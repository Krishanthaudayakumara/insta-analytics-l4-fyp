import os
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Test the basic functionality step by step
try:
    import sys
    sys.path.append('src')
    print("Path added successfully")
    
    from preprocessing.clustered_data_processor import ClusteredDataProcessor
    print("Import successful")
    
    processor = ClusteredDataProcessor()
    print("Processor created")
    
    clusters = processor.get_cluster_names()
    print("Clusters found:", clusters)
    
    if clusters:
        print(f"Testing with cluster: {clusters[0]}")
        
        # Check if the cluster directory exists
        cluster_path = os.path.join(processor.base_path, clusters[0])
        print(f"Cluster path: {cluster_path}")
        print(f"Path exists: {os.path.exists(cluster_path)}")
        
        if os.path.exists(cluster_path):
            user_folders = [d for d in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, d))]
            print(f"Found {len(user_folders)} user folders")
            
            if user_folders:
                # Test with first user folder
                test_user = user_folders[0]
                user_path = os.path.join(cluster_path, test_user)
                info_files = [f for f in os.listdir(user_path) if f.endswith('.info')]
                print(f"User {test_user} has {len(info_files)} .info files")
                
                if info_files:
                    # Test reading one file
                    test_file = os.path.join(user_path, info_files[0])
                    print(f"Testing file: {test_file}")
                    
                    with open(test_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print("Successfully read JSON file")
                        print("Keys in JSON:", list(data.keys())[:10])  # First 10 keys
                        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
