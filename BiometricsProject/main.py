import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Suppress all unnecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

try:
    from src.signature_verifier import WorldClassSignatureVerificationSystem
    from src.utils import create_sample_dataset, load_images_from_folder
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed.")
    sys.exit(1)


def setup_directories():
    """Create necessary directories"""
    directories = ['data/references', 'data/test', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def clear_old_references():
    """Clear old reference file to start fresh with new algorithm"""
    if os.path.exists('reference_signatures.json'):
        backup_name = f'reference_signatures_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.rename('reference_signatures.json', backup_name)
        print(f"⚠️  Old references backed up as: {backup_name}")
        print("🔄 Starting fresh with world-class algorithm")
        return True
    return False


def main():
    """Main application function"""
    print("=== WORLD-CLASS SIGNATURE VERIFICATION SYSTEM ===")
    print("Using advanced multi-level verification with stroke analysis\n")

    # Setup directories
    setup_directories()

    # Clear old references to start fresh
    cleared = clear_old_references()

    # Initialize world-class verification system
    verifier = WorldClassSignatureVerificationSystem(similarity_threshold=0.85)

    # Try to load existing references (only if we didn't just clear them)
    if not cleared and os.path.exists('reference_signatures.json'):
        try:
            # Temporarily reduce logging level
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)

            loaded = verifier.load_references()

            # Restore logging level
            logging.getLogger().setLevel(original_level)

            if loaded:
                print(f"✓ Loaded {len(verifier.get_reference_list())} reference signatures")
            else:
                print("Starting with fresh references database.")
        except Exception as e:
            print(f"Starting with fresh references due to loading error.")
    else:
        print("No existing references found. Starting fresh.")

    while True:
        print("\n" + "=" * 50)
        print("1. Add Reference Signature")
        print("2. Test Signature")
        print("3. List Reference Signatures")
        print("4. View History")
        print("5. Settings")
        print("6. Create Sample Dataset")
        print("7. Exit")
        print("=" * 50)

        try:
            choice = input("\nSelect option (1-7): ").strip()

            if choice == '1':
                name = input("Enter signature name: ").strip()
                if not name:
                    print("Error: Signature name cannot be empty!")
                    continue

                path = input("Enter image path: ").strip()

                # Remove quotes if user included them
                path = path.strip('"').strip("'")

                print(f"Looking for file: {path}")
                print(f"File exists: {os.path.exists(path)}")

                if not os.path.exists(path):
                    print("Error: File does not exist!")
                    print("Tip: Try using full path or move image to project folder")
                    continue

                success = verifier.add_reference_signature(name, path)
                if success:
                    print(f"✓ Reference signature '{name}' added successfully!")
                else:
                    print("✗ Failed to add reference signature")

            elif choice == '2':
                if not verifier.reference_signatures:
                    print("No reference signatures available. Please add some first.")
                    continue

                print("Available references:", ", ".join(verifier.get_reference_list()))
                ref_name = input("Enter reference name: ").strip()

                if ref_name not in verifier.reference_signatures:
                    print("Error: Reference name not found!")
                    continue

                test_path = input("Enter test image path: ").strip()
                test_path = test_path.strip('"').strip("'")

                if not os.path.exists(test_path):
                    print("Error: Test file does not exist!")
                    continue

                print("Running signature verification...")
                similarity, is_genuine, test_features, verification_message = verifier.compare_signatures(test_path,
                                                                                                          ref_name)

                if similarity is not None:
                    print(f"\n" + "=" * 50)
                    print(f"SIGNATURE VERIFICATION RESULT:")
                    print(f"Similarity Score: {similarity:.3f}")
                    print(f"Threshold: {verifier.similarity_threshold}")
                    print(f"Verdict: {'GENUINE' if is_genuine else 'FORGED'}")
                    print(f"Verification: {verification_message}")
                    print("=" * 50)

                    viz = input("\nShow detailed visualization? (y/n): ").lower()
                    if viz == 'y' and test_features is not None:
                        verifier.visualize_comparison(test_path, ref_name, similarity, is_genuine, test_features,
                                                      verification_message)
                else:
                    print("Error: Could not verify signature")

            elif choice == '3':
                references = verifier.get_reference_list()
                if references:
                    print("\nReference Signatures:")
                    for i, ref in enumerate(references, 1):
                        print(f"{i}. {ref}")
                else:
                    print("No reference signatures available")

            elif choice == '4':
                history = verifier.get_verification_history()
                if history:
                    print("\nVerification History (last 10):")
                    for i, record in enumerate(history[-10:], 1):
                        print(f"{i}. {record['timestamp'][11:19]} - {record['reference_name']} - "
                              f"Score: {record['similarity_score']:.3f} - {record['verdict']}")
                else:
                    print("No verification history")

            elif choice == '5':
                new_threshold = input(
                    f"Current threshold: {verifier.similarity_threshold}. Enter new threshold (0.7-0.95): ")
                try:
                    new_threshold = float(new_threshold)
                    if 0.7 <= new_threshold <= 0.95:
                        verifier.similarity_threshold = new_threshold
                        print(f"✓ Threshold updated to {new_threshold}")
                    else:
                        print("Threshold must be between 0.7 and 0.95")
                except ValueError:
                    print("Invalid threshold value")

            elif choice == '6':
                create_sample_dataset()
                print("Sample dataset structure created!")

            elif choice == '7':
                verifier.save_references()
                print("References saved. Goodbye!")
                break

            else:
                print("Invalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()