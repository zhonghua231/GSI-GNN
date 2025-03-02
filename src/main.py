import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_name}: {e}")

def main():

    scripts = [
        'DataLoader.py',
        'GSIGNN.py',
        'SiameseNetwork.py'
    ]
    for script in scripts:
        run_script(script)

if __name__ == '__main__':
    main()
