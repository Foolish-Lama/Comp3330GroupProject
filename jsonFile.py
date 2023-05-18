import json
import os

class json_file():
    # stores a list of objects, automatically keeps track of id
    def __init__ (self, path):
        self.path = path
        self.content = []
        self.id_counter = 0

        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.content = json.loads(f.read())
                self.id_counter = len(self.content)
        else:
            with open(self.path, "w") as f:
                self.write()

    def write(self):
        with open(self.path, "w+") as f:
            json.dump(self.content, f, indent=4)

    def add(self, new_entry):
        self.id_counter += 1
        new_entry["id"] = self.id_counter
        self.content.append(new_entry)
        self.write()
        return new_entry

    def get(self, id):
        for e in self.content:
            if e["id"] == id:
                return e
        raise Exception("entry does not exist with id: " + str(id))

    def eupdate(self, id, updates):
        entry = self.get(id)
        entry.update(updates)
        self.write()
        return entry
    
    def delete(self, id):
        entry = self.get(id)
        del self.content[self.content.index(entry)]
        self.write()

    def find(self, key, value):
        found = []
        for e in self.content:
            if e[key] == value:
                found.append(e)
        return found

