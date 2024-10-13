import asyncio
import ray
import logging
import random
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
import litellm
from enum import Enum
from ray.util.actor_pool import ActorPool
import time
import threading
import agentics

# Layered System for a Scalable, Dynamic Multi-Agent Environment

# Initialize Ray for distributed task execution
ray.init()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgentSystem")

# Enum for message types, facilitating better communication protocols
class MessageType(Enum):
    TASK = 1
    RESPONSE = 2
    BROADCAST = 3
    CONSENSUS_PROPOSAL = 4
    CONSENSUS_VOTE = 5

# FastAPI application setup
app = FastAPI()

# Static files and templates for enhanced web interface
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# HTML for WebSocket client integrated with the agentics web interface
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Agentic AI Multi-Agent Streaming</title>
        <link rel="stylesheet" type="text/css" href="/static/styles.css">
        <script src="/static/agentics.js"></script>
    </head>
    <body>
        <h1>Agent Supervision Interface</h1>
        <textarea id="log" rows="20" cols="100"></textarea><br>
        <button onclick="sendMessage()">Envoyer Commande</button>
        <input id="commandInput" type="text"/>
        <script>
            let ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                let log = document.getElementById('log');
                log.value += "\n" + event.data;
            };
            function sendMessage() {
                let input = document.getElementById("commandInput");
                ws.send(input.value);
                input.value = "";
            }
        </script>
        <div id="agentics-dashboard"></div>
        <script>
            // Initialize the Agentics dashboard
            agentics.initDashboard("agentics-dashboard", ws);
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await process_human_command(data, websocket)
        await websocket.send_text(f"Commande humaine reçue: {data}")

# LiteLLM query handler
def query_llm(prompt: str) -> str:
    try:
        response = litellm.completion(prompt=prompt)
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la requête LLM: {str(e)}")
        return f"Erreur: {str(e)}"

@ray.remote
class Agent:
    def __init__(self, agent_id, neighbors: List[int]):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.task_queue = []
        self.memory = {}  # Local memory to store past solutions
        self.state = "idle"
        self.role = None  # Role attribute for specialized agents

    def assign_role(self, role):
        self.role = role
        logger.info(f"Agent {self.agent_id} assigned role: {role}")

    def act(self):
        # Role-specific behavior
        if self.role == 'explorer':
            # Logic for explorer agents (e.g., data gathering)
            new_data = agentics.explore_environment()  # Updated to use agentics library for exploration
            self.broadcast_message({'type': MessageType.BROADCAST, 'content': new_data})
        elif self.role == 'analyzer':
            # Logic for analyzer agents (e.g., data processing)
            if self.task_queue:
                data_to_analyze = self.task_queue.pop(0)
                analysis_result = agentics.analyze_data(data_to_analyze)  # Using agentics for analysis
                self.state = f"analyzing data: {analysis_result}"
                logger.info(f"Agent {self.agent_id} analyzed data: {analysis_result}")
        elif self.role == 'communicator':
            # Logic for communicator agents (e.g., facilitating communication)
            if self.task_queue:
                message = self.task_queue.pop(0)
                self.broadcast_message({'type': MessageType.RESPONSE, 'content': message})
                logger.info(f"Agent {self.agent_id} broadcasted message: {message}")

        # Update Agentics dashboard with agent state
        agentics.updateAgentState(self.agent_id, self.state)

        # General task execution
        if self.task_queue:
            current_task = self.task_queue.pop(0)
            self.state = f"working on task {current_task}"
            logger.info(f"Agent {self.agent_id} commence à travailler sur la tâche: {current_task}")
        else:
            self.state = "idle"
        return f"Agent {self.agent_id} is {self.state}"

    def receive_message(self, message: Dict):
        message_type = message['type']
        if message_type == MessageType.CONSENSUS_PROPOSAL:
            logger.info(f"Agent {self.agent_id} reçoit une proposition de consensus: {message['content']}")
            # Dummy voting mechanism
            vote = random.choice([True, False])
            self.send_vote(message['sender'], vote)
        elif message_type == MessageType.TASK:
            logger.info(f"Agent {self.agent_id} reçoit une tâche: {message['content']}")
            self.task_queue.append(message['content'])

    def send_vote(self, to_agent_id, vote: bool):
        agents[to_agent_id].receive_message.remote({
            'type': MessageType.CONSENSUS_VOTE,
            'sender': self.agent_id,
            'vote': vote
        })

    async def collaborate(self):
        collective_state = []
        for neighbor in self.neighbors:
            state = await agents[neighbor].get_state.remote()
            collective_state.append(state)
        all_memories = ray.get([agent.get_memory.remote() for agent in agents])
        merged_memory = query_llm(f"Merge the following memories: {all_memories}")
        self.memory.update(merged_memory)
        merged_solution = query_llm(f"Merge the following states: {', '.join(collective_state)}")
        self.state = f"collaborating and merging solutions: {merged_solution}"
        logger.info(f"Agent {self.agent_id} a collaboré avec ses voisins. Nouvel état: {self.state}")
        agentics.updateAgentState(self.agent_id, self.state)  # Update Agentics dashboard with collaboration result
        return f"Agent {self.agent_id} collaborated with neighbors. New state: {self.state}"

    def get_state(self):
        return self.state

    def get_memory(self):
        return self.memory

    def assign_task(self, task: str):
        self.task_queue.append(task)
        logger.info(f"Agent {self.agent_id} a reçu une nouvelle tâche: {task}")
        return f"Agent {self.agent_id} assigned new task: {task}"

# Create agents dynamically, with specializations and shared memory
agent_count = int(os.getenv("AGENT_COUNT", 5))
agents = [Agent.remote(i, [j for j in range(agent_count) if j != i]) for i in range(agent_count)]
agent_pool = ActorPool(agents)

# Define roles for agents to specialize their functions
roles = ['explorer', 'analyzer', 'communicator']
for i, agent in enumerate(agents):
    role = roles[i % len(roles)]
    ray.get(agent.assign_role.remote(role))

# Function to handle human-in-the-loop commands
async def process_human_command(command: str, websocket: WebSocket):
    command_parts = command.split(maxsplit=1)
    if len(command_parts) < 2:
        await websocket.send_text("Commande non valide. Veuillez fournir une description de la tâche.")
        return
    command_type, task_description = command_parts
    if command_type == "assign":
        target_agent_id = random.randint(0, agent_count - 1)
        result = agents[target_agent_id].assign_task.remote(task_description)
        await websocket.send_text(ray.get(result))
    elif command_type == "collaborate":
        results = await asyncio.gather(*[agent.collaborate.remote() for agent in agents])
        for result in results:
            await websocket.send_text(ray.get(result))
    else:
        await websocket.send_text("Commande inconnue. Veuillez réessayer.")
