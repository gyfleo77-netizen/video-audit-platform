from backend import app, db


if __name__ == "__main__":
    with app.app_context():

        db.create_all()
        print("????????")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
