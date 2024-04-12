from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import prediction

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Test Routes to check Server is runing
@app.get("/", tags=["test"], description="To check if server is runing")
async def root():
    return {"message": "Server Runing!"}

#Including all Router
app.include_router(prediction.router)
 
