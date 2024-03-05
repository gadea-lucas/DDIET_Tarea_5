# Make GET request to the web
import requests

# Extract content and save it to a file
def get_data(urls, path):
    for url in urls:
        response = requests.get(url)
        content = response.text  # full HTML
        
        # Save to a file
        filename = path + 'webpage_' + url.split('/')[-1].replace('%', '_').replace('?', '_') + ".txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"[INFO] Contenido de {url} guardado en {filename}")

def main():
    # Webpages URL
    urls_es = [
        "https://es.wikipedia.org/wiki/Solanum_tuberosum",
        "https://poemas.uned.es/poema/palabras-para-julia-jose-agustin-goytisolo",
        "https://www.pequerecetas.com/receta/tortilla-de-patatas",
        "https://www.ajedrezeureka.com/reglas-basicas-del-ajedrez",
        "https://docs.python.org/es/3/installing/index.html"
    ]
    urls_en = [
        "https://en.wikipedia.org/wiki/Potato",
        "https://genius.com/The-beatles-here-comes-the-sun-lyrics",
        "https://littlespoonfarm.com/apple-pie-recipe",
        "https://www.chessjournal.com/how-to-play-chess",
        "https://docs.python.org/3/installing/index.html"
    ]
    # Extract content and save it to a file
    get_data(urls_es, "./colecciones/web_es/")
    get_data(urls_en, "./colecciones/web_en/")


if __name__ == "__main__":
    main()
        
