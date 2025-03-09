from roboflow import Roboflow

# Configura tu API key de Roboflow
API_KEY = "MRdWG5zr7plOxlPk4ZkU"

# Inicializa Roboflow
print("Inicializando Roboflow...")
rf = Roboflow(api_key=API_KEY)

# Obtener workspace
print("Obteniendo información del workspace...")
workspace = rf.workspace()

# Listar proyectos disponibles
print("\nProyectos disponibles:")
projects = workspace.projects()

if not projects:
    print("No se encontraron proyectos en tu workspace.")
else:
    for i, project in enumerate(projects):
        print(f"{i+1}. {project}")
    
    # Para el primer proyecto, obtener más detalles
    print("\nObteniendo detalles del primer proyecto...")
    try:
        first_project = workspace.project(projects[0])
        print(f"Project Name: {first_project.name if hasattr(first_project, 'name') else 'No disponible'}")
        print(f"Project Type: {first_project.type if hasattr(first_project, 'type') else 'No disponible'}")
        
        # Listar versiones del modelo
        print("\nVersiones disponibles:")
        try:
            versions = first_project.versions()
            if versions:
                for version in versions:
                    print(f"- Versión {version}")
            else:
                print("No se encontraron versiones para este proyecto.")
        except Exception as e:
            print(f"Error al obtener versiones: {e}")
    except Exception as e:
        print(f"Error al obtener detalles del proyecto: {e}")