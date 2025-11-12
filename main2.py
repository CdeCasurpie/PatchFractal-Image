"""
main2.py - Punto de entrada principal
Permite elegir entre generar un proyecto fractal o renderizar uno existente
"""

import sys
import os

def print_menu():
    print("\n" + "=" * 60)
    print("  FRACTAL GENERATOR & VIEWER - v7.0")
    print("=" * 60)
    print("  1. Generar nuevo proyecto fractal")
    print("  2. Renderizar proyecto existente")
    print("  3. Salir")
    print("=" * 60)

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "generate":
            if len(sys.argv) < 3:
                print("Uso: python main2.py generate <imagen.jpg> [nombre_proyecto]")
                sys.exit(1)
            
            import generate
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            
            image_path = sys.argv[2]
            project_name = sys.argv[3] if len(sys.argv) > 3 else None
            generate.generate_fractal(image_path, project_name)
            
        elif mode == "render":
            import render
            project_name = sys.argv[2] if len(sys.argv) > 2 else None
            
            if project_name:
                render.render_project(project_name)
            else:
                project_name = render.select_project()
                if project_name:
                    render.render_project(project_name)
        else:
            print(f"Modo desconocido: {mode}")
            print("Usa: python main2.py [generate|render]")
            sys.exit(1)
    else:
        # Modo interactivo
        while True:
            print_menu()
            choice = input("\nSelecciona una opción (1-3): ").strip()
            
            if choice == "1":
                image_path = input("\nRuta de la imagen: ").strip()
                
                if not os.path.exists(image_path):
                    print(f"❌ No se encuentra el archivo: {image_path}")
                    continue
                
                project_name = input("Nombre del proyecto (Enter para auto): ").strip()
                project_name = project_name if project_name else None
                
                import generate
                import multiprocessing
                multiprocessing.set_start_method('spawn', force=True)
                
                try:
                    generate.generate_fractal(image_path, project_name)
                    
                    render_now = input("\n¿Renderizar ahora? (s/n): ").strip().lower()
                    if render_now == 's':
                        import render
                        # Usar el nombre del proyecto generado
                        if project_name is None:
                            basename = os.path.splitext(os.path.basename(image_path))[0]
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            project_name = f"{basename}_{timestamp}"
                        render.render_project(project_name)
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == "2":
                import render
                project_name = render.select_project()
                if project_name:
                    try:
                        render.render_project(project_name)
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
            
            elif choice == "3":
                print("\n¡Hasta luego! 👋")
                break
            
            else:
                print("❌ Opción inválida")

if __name__ == "__main__":
    main()