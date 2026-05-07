import importlib
from typing import Tuple
from agents.base_agent import ModelProvider

def parse_agent_spec(agent_spec: str) -> Tuple[str, str]:
    """
    解析 Agent 规范字符串。

    Args:
        agent_spec: 形如 "module.path:ClassName" 的字符串

    Returns:
        (module_name, class_name) 的二元组
    """
    if ':' not in agent_spec:
        raise ValueError(
            f"Invalid agent specification: {agent_spec}. "
            f"Expected format: 'module.path:ClassName'"
        )

    module_name, class_name = agent_spec.split(':', 1)
    return module_name, class_name


def load_agent(agent_spec: str, api_key: str, base_url: str) -> ModelProvider:
    """
    动态加载 Agent 实现。

    Args:
        agent_spec: Agent 规范 "module.path:ClassName"
        api_key: 接口密钥
        base_url: 接口基础地址

    Returns:
        Agent 实例
    """
    module_name, class_name = parse_agent_spec(agent_spec)
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    return agent_class(api_key=api_key, base_url=base_url)