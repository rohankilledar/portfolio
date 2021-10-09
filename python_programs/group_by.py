def group_by_owners(files):
    answer = dict()
    
    dup_owners = [owner for owner in files.values()]
    owners=[]
    for owner in dup_owners:
        if owner not in owners:
            owners.append(owner)
    for owner in owners:
        fileList = []
        for file, ow in files.items():
            if owner == ow:
                fileList.append(file)
        answer[owner] = fileList
    
    
    return answer

if __name__ == "__main__":    
    files = {
        'Input.txt': 'Randy',
        'Code.py': 'Stan',
        'Output.txt': 'Randy'
    }   
    print(group_by_owners(files))