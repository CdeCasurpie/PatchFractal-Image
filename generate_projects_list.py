#!/usr/bin/env python3
"""
generate_projects_list.py
Genera un archivo projects.json con la lista de proyectos disponibles.
Ejecuta este script cada vez que agregues un nuevo proyecto.
"""
import os
import json

OUTPUT_DIR = 'output'
PROJECTS_FILE = 'output/projects.json'

def get_projects():
    """Obtiene la lista de proyectos en la carpeta output/"""
    if not os.path.exists(OUTPUT_DIR):
        return []
    
    projects = []
    for item in os.listdir(OUTPUT_DIR):
        project_path = os.path.join(OUTPUT_DIR, item)
        # Solo directorios que tienen project.json
        if os.path.isdir(project_path):
            project_json = os.path.join(project_path, 'project.json')
            if os.path.exists(project_json):
                projects.append(item)
    
    return sorted(projects)

def main():
    projects = get_projects()
    
    # Guardar la lista
    with open(PROJECTS_FILE, 'w') as f:
        json.dump({'projects': projects}, f, indent=2)
    
    print(f"✅ Generado {PROJECTS_FILE} con {len(projects)} proyectos:")
    for p in projects:
        print(f"   - {p}")

if __name__ == '__main__':
    main()
