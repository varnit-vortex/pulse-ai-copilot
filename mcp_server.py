from mcp.server.fastmcp import FastMCP
from tools import check_patient_triage_status
mcp = FastMCP('PulseAI-Healthcare-Tools')

@mcp.tool()
def check_patient_triage(record_id: str) -> dict:
    return check_patient_triage_status(record_id)
if __name__ == '__main__':
    mcp.run()