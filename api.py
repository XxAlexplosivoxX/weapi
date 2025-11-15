import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import face_recognition
from load_images import known_face_encodings, known_face_names

app = FastAPI() # acá se crea la instancia de FastAPI

@app.post("/verificar-careta") # ruta para verificar rostro
async def process_image(image: UploadFile = File(...)): # en la función como parámetro se espera una imagen como archivo
    try:
        # se carga la imagen para el face_recognition
        in_face_image = face_recognition.load_image_file(
            image.file
        )

        # se obtienen las codificaciones (en una lista por cada rostro detectado) de la cara en la imagen subida
        in_face_encodings = face_recognition.face_encodings(in_face_image)
        
        # si no se detecta ninguna cara, se lanza un error
        if len(in_face_encodings) == 0:
            return JSONResponse(status_code=400, content={"msg": f"Pero poné la cara, cagón", "valid": False})
        # si se detectan más de una sola cara, se lanza un error
        if len(in_face_encodings) > 1:
            return JSONResponse(status_code=400, content={"msg": f"Una cara a la vez, no {len(in_face_encodings)} fokin caras ctm", "valid": False})

        # se comparan las codificaciones de la cara subida con las conocidas
        matches = face_recognition.compare_faces(
            known_face_encodings,
            in_face_encodings[0]
        )

        # se verifica si hubo alguna coincidencia
        if True in matches:
            idx = matches.index(True)
            name = known_face_names[idx]
            # si hay coincidencia, se retorna el nombre asociado
            return JSONResponse(content={"msg": f"Te conozco wn qliao, eres {name}", "valid": True})
        else:
            # si no hay coincidencia, se indica que no se conoce al qliao de la imagen
            return JSONResponse(content={"msg": "No te conozco wn qliao", "valid": False})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"msg": f"Algo falló pibe, sorry: {str(e)}", "valid": False},
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=False
    )
