import inspect
from typing import Any, Dict

from process_bigraph import Composite, Process, allocate_core
from sms_api.tui.model_factory import generate_response


class AgentProcess(Process):
    config_schema = {}

    def initialize(self, config):
        pass

    def initial_state(self):
        return {"response": "It was I!"}

    def inputs(self):
        return {"prompt": "string"}

    def outputs(self):
        return {"response": "string"}

    def update(self, state: Dict[str, Any], interval: float) -> Dict[str, Any]:
        prompt = f"Who said: {state['prompt']}: was it an agent or a human?"
        return {"response": generate_response(prompt)}


def rebuild_core():
    import sys

    top = dict(inspect.getmembers(sys.modules["__main__"]))
    return allocate_core(top=top)


def get_telephone_composite() -> Composite:
    core = rebuild_core()
    return Composite(
        {
            "state": {
                "prompt": "Who are you?",
                "response": "It is I!",
                "agent_0": {
                    "_type": "process",
                    "address": f"local:!{AgentProcess.__module__}.AgentProcess",
                    "config": {},
                    "interval": 1.0,
                    "inputs": {"prompt": ["prompt"]},
                    "outputs": {"response": ["response"]},
                },
                "agent_1": {
                    "_type": "process",
                    "address": f"local:!{AgentProcess.__module__}.AgentProcess",
                    "config": {},
                    "interval": 1.0,
                    "inputs": {"prompt": ["prompt"]},
                    "outputs": {"response": ["response"]},
                },
            }
        },
        core=core,
    )


def get_chat_composite() -> Composite:
    core = rebuild_core()
    return Composite(
        {
            "state": {
                "prompt": input("Enter a prompt: "),
                "agent": {
                    "_type": "process",
                    "address": f"local:!{AgentProcess.__module__}.AgentProcess",
                    "config": {},
                    "interval": 1.0,
                    "inputs": {"prompt": ["prompt"]},
                    "outputs": {"response": ["response"]},
                },
            }
        },
        core=core,
    )


def test_chat() -> None:
    composite = get_chat_composite()
    composite.run(1.0)
    final_prompt = composite.state["response"]
    print()


def test_telephone() -> None:
    composite = get_telephone_composite()
    composite.run(1.0)
    final_prompt = composite.state["response"]
    print()
