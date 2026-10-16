import subprocess
import sys
import os

def run_pipeline(scenario_filename):
    print(f"\n==============================================")
    print(f" RUNNING PIPELINE FOR: {scenario_filename}")
    print(f"==============================================")
    
    cmd = [sys.executable, "run.py", "--scenario", scenario_filename, "--seed", "42", "--out", "./"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR / ERRORS:")
        print(result.stderr)
        
    return result.returncode == 0

def main():
    success_r = run_pipeline("scenario_R.json")
    success_h = run_pipeline("scenario_H.json")
    
    print(f"\n==============================================")
    print(f" VERIFICATION SUMMARY")
    print(f"==============================================")
    print(f"Scenario R Pipeline: {'SUCCESS' if success_r else 'FAILED'}")
    print(f"Scenario H Pipeline: {'SUCCESS' if success_h else 'FAILED'}")
    
    artifacts = ["qubo.json", "circuit.qasm", "counts.json", "schedule.json"]
    print("\nChecking generated output artifacts:")
    for art in artifacts:
        exists = os.path.exists(art)
        size = os.path.getsize(art) if exists else 0
        print(f" - {art}: {'EXISTS (' + str(size) + ' bytes)' if exists else 'MISSING'}")

if __name__ == "__main__":
    main()