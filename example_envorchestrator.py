from veca.env_manager import EnvOrchestrator
import argparse

if __name__ == "__main__":
    # Executing the Environment Orchestrator process at another server.
    parser = argparse.ArgumentParser(description='VECA Navigation')
    parser.add_argument('--port', type=int, required= True)
    parser.add_argument('--localport', type=int, required= True)
    args = parser.parse_args()
    env = EnvOrchestrator(
        ip = "147.46.240.40",   # ip and port of remote Envionment Orchestrator master
        port = args.port,        # Exposed port of Environment Orchestrator
        port_instance = args.localport   # inter-process communication port with orchestrator and unity instances (localport ~ localport + num_envs)
        )
