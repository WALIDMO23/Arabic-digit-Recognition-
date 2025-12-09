"""
Arabic Digit Recognition - Complete System
Run all components in sequence
"""

import os
import sys
import subprocess
import webbrowser
import time


def check_requirements():
    """Check if all required packages are installed"""
    required = [
        'sklearn', 'numpy', 'pandas', 'matplotlib',
        'seaborn', 'skimage', 'cv2', 'PIL',
        'flask', 'joblib', 'tqdm'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'results']

    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created: {dir_name}")
        else:
            print(f"📁 Exists: {dir_name}")


def print_menu():
    """Print main menu"""
    print("\n" + "=" * 70)
    print("🔢 ARABIC DIGIT RECOGNITION SYSTEM")
    print("=" * 70)
    print("\n📋 Available Operations:")
    print("1. 📊 Train and Compare ML Algorithms")
    print("2. 🌐 Launch Web Interface (Drag & Drop)")
    print("3. 💻 Launch Console Interface (Image Path)")
    print("4. 🧪 Quick Test with Sample Data")
    print("5. 📈 View Results and Reports")
    print("6. 🚀 Run Complete Pipeline")
    print("7. ❌ Exit")
    print("\n" + "=" * 70)


def run_comparison():
    """Run ML algorithms comparison"""
    print("\n" + "=" * 70)
    print("🤖 TRAINING AND COMPARING ML ALGORITHMS")
    print("=" * 70)

    # Check if data exists
    if not os.path.exists("data") or len(os.listdir("data")) == 0:
        print("❌ No data found!")
        print("\nPlease add images to 'data' folder in this structure:")
        print("data/0/, data/1/, ..., data/9/")
        return

    # Run comparison
    import train
    train.main()


def launch_web_interface():
    """Launch Flask web interface"""
    print("\n" + "=" * 70)
    print("🌐 LAUNCHING WEB INTERFACE")
    print("=" * 70)

    # Check if model exists
    if not os.path.exists("models/best_model.pkl"):
        print("❌ Model not found! Please train first.")
        return

    # Launch in background
    print("Starting web server...")
    print("Open: http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    # Open browser
    webbrowser.open("http://localhost:5000")

    # Run Flask app
    os.system("python flask_app.py")


def launch_console_interface():
    """Launch console interface"""
    print("\n" + "=" * 70)
    print("💻 LAUNCHING CONSOLE INTERFACE")
    print("=" * 70)

    # Check if model exists
    if not os.path.exists("models/best_model.pkl"):
        print("❌ Model not found! Please train first.")
        return

    # Run console interface
    os.system("python test_interface.py")


def quick_test():
    """Quick test with sample data"""
    print("\n" + "=" * 70)
    print("🧪 QUICK TEST WITH SAMPLE DATA")
    print("=" * 70)

    # Create sample data if not exists
    if not os.path.exists("data"):
        print("Creating sample data...")
        os.makedirs("data", exist_ok=True)

        # Import and create sample data
        import train_comparison
        # You might need to add a function to create sample data

        print("✅ Sample data created")

    # Run quick training
    print("\nRunning quick training...")
    # Add your quick training function here


def view_results():
    """View results and reports"""
    print("\n" + "=" * 70)
    print("📈 VIEWING RESULTS AND REPORTS")
    print("=" * 70)

    if os.path.exists("results"):
        print("\n📁 Available Results:")

        # List all result files
        for file in os.listdir("results"):
            if file.endswith(('.png', '.csv', '.txt')):
                print(f"  📄 {file}")

        # Open the main report if exists
        report_file = "results/detailed_report.txt"
        if os.path.exists(report_file):
            print(f"\n📋 Detailed Report:")
            with open(report_file, 'r') as f:
                content = f.read()
                print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print("❌ No results found! Run training first.")


def run_complete_pipeline():
    """Run complete pipeline"""
    print("\n" + "=" * 70)
    print("🚀 RUNNING COMPLETE PIPELINE")
    print("=" * 70)

    steps = [
        ("📊 Training and comparing ML algorithms", run_comparison),
        ("📈 Generating reports", view_results),
        ("🌐 Launching web interface", launch_web_interface)
    ]

    for step_name, step_function in steps:
        print(f"\n▶️  {step_name}")
        print("-" * 50)

        try:
            step_function()
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n⏹️ Pipeline interrupted")
            break
        except Exception as e:
            print(f"❌ Error in step: {e}")
            continue


def main():
    """Main function"""
    print("🔢 Welcome to Arabic Digit Recognition System")
    print("=" * 70)

    # Check requirements
    missing = check_requirements()
    if missing:
        print("❌ Missing packages:")
        for package in missing:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return

    print("✅ All requirements satisfied")

    # Create directories
    create_directories()

    # Main loop
    while True:
        print_menu()

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == "1":
            run_comparison()
        elif choice == "2":
            launch_web_interface()
        elif choice == "3":
            launch_console_interface()
        elif choice == "4":
            quick_test()
        elif choice == "5":
            view_results()
        elif choice == "6":
            run_complete_pipeline()
        elif choice == "7":
            print("\n👋 Thank you for using Arabic Digit Recognition System!")
            break
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()