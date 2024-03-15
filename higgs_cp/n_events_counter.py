import os
import ROOT

def count_events(root_file_path, tree_name="Events"):
    # Open the ROOT file
    root_file = ROOT.TFile(root_file_path)

    # Check if the file is open successfully
    if not root_file.IsOpen():
        print("Error: Unable to open file {}".format(root_file_path))
        return -1

    # Access the TTree
    tree = root_file.Get(tree_name)

    # Check if the tree is found
    if not tree:
        print("Error: TTree '{}' not found in file {}".format(tree_name, root_file_path))
        root_file.Close()
        return -1

    # Get the number of entries/events in the tree
    num_entries = tree.GetEntries()

    # Close the ROOT file
    root_file.Close()

    return num_entries

def count_events_in_folder(folder_path, tree_name="Events"):
    total_events = 0
    total_files = 0

    # Iterate through all files in the folder
    for root_file in os.listdir(folder_path):
        if root_file.endswith(".root"):
            total_files += 1
            file_path = os.path.join(folder_path, root_file)
            num_events = count_events(file_path, tree_name)
            
            if num_events >= 0:
                print("Number of events in {}: {}".format(root_file, num_events))
                total_events += num_events
            else:
                print("Failed to count events for {}".format(root_file))

    return total_files, total_events

def main():
    # Folder containing ROOT files
    folder_path = "/eos/cms/store/group/phys_tau/irandreo/Run3_22/EGamma_Run2022C"
    
    # Name of the TTree in the ROOT files
    tree_name = "Events"

    # Count events and files in the folder
    total_files, total_events = count_events_in_folder(folder_path, tree_name)

    print("\nTotal number of ROOT files in the folder: {}".format(total_files))
    print("Total number of events in the folder: {}".format(total_events))

if __name__ == "__main__":
    main()
