from api.factory import create_app
#import uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)

