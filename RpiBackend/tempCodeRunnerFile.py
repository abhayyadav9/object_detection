import app

if __name__ == "__main__":
    print("\n🔥 YOLOv10 Pose Server Running at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)