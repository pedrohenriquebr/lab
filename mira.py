import tkinter as tk

# ==========================================
# AJUSTE AQUI IGUAL AO SEU SCRIPT DE TREINO
# ==========================================
MONITOR = {"top": 100, "left": 500, "width": 600, "height": 600} 
# ==========================================

def criar_mira():
    root = tk.Tk()
    
    # Remove a barra de título e bordas do Windows
    root.overrideredirect(True)
    
    # Mantém a janela sempre no topo
    root.wm_attributes("-topmost", True)
    
    # Define a cor de transparência (Windows)
    # Tudo que for "white" na janela ficará invisível
    root.wm_attributes("-transparentcolor", "white")
    
    # Configura geometria (Tamanho + Posição X + Posição Y)
    geo_str = f"{MONITOR['width']}x{MONITOR['height']}+{MONITOR['left']}+{MONITOR['top']}"
    root.geometry(geo_str)
    
    # Cria um Canvas (área de desenho)
    canvas = tk.Canvas(root, bg="white", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Desenha o Retângulo Vermelho (Borda grossa)
    # Coordenadas: x1, y1, x2, y2
    canvas.create_rectangle(
        0, 0, 
        MONITOR['width']-2, MONITOR['height']-2, 
        outline="red", 
        width=5
    )
    
    # Botão para fechar (pequeno no canto)
    btn = tk.Button(root, text="X", command=root.destroy, bg="red", fg="white")
    btn.place(x=0, y=0)
    
    print(f"🎯 Mira ativa em: {MONITOR}")
    print("Feche a janela clicando no 'X' vermelho para encerrar.")
    
    root.mainloop()

if __name__ == "__main__":
    criar_mira()