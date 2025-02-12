import os

def combine_source_files(project_dir, output_file, extensions=['.py']):
    """
    Combine all source files with specified extensions into a single file.
    
    Args:
        project_dir (str): Root directory of the project
        output_file (str): Path to output file
        extensions (list): List of file extensions to include
    """
    # Clear/create output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        pass  # Just create/clear the file

    # Walk through directory
    for root, _, files in os.walk(project_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    outfile.write(f"\n-- {file_path} -- \n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"\nError reading file: {str(e)}\n")
                    outfile.write(" \n")

    print(f"Files combined into {output_file}")

if __name__ == "__main__":
    # Set your paths here
    project_directory = "."  # Current directory
    output_file = "combined_source.txt"
    extensions = ['.py']  # Add more extensions if needed

    combine_source_files(project_directory, output_file, extensions)