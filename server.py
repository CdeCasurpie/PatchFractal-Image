"""
Simple HTTP server para servir el index.html y los proyectos
"""
import http.server
import socketserver
import os

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # CORS headers para desarrollo
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

os.chdir('/home/cesar/Escritorio/fractalInfinito')

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"=" * 60)
    print(f"  🚀 Servidor HTTP iniciado en http://localhost:{PORT}")
    print(f"=" * 60)
    print(f"  📂 Sirviendo desde: {os.getcwd()}")
    print(f"  🌐 Abre en tu navegador: http://localhost:{PORT}")
    print(f"=" * 60)
    print("  Presiona Ctrl+C para detener el servidor\n")
    httpd.serve_forever()
