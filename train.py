def train():
    """
    Train the model using the provided training data and configuration.
    """
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # Load the standard YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")

    print("Standard YOLOv8 segmentation model loaded successfully.")

    # Start training
    # Typically the dataset will contain a data.yaml file
    model.train(
        data="dataset/Pothole_Segmentation_YOLOv8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        device=device,
    )

    dummy_input = torch.randn(1, 3, 640, 640)

    dynamic_input_shapes = (
        {
            0: torch.export.Dim("batch_size"),
        },
    )

    # Define output names for clarity in the ONNX graph
    output_names_list = ["detection", "prototype"]

    # disable batch normalizagtion
    model.model.eval()

    torch.onnx.export(
        model=model.model,
        args=dummy_input,  # Provide a sample input tensor
        f="model.onnx",
        opset_version=17,
        input_names=["input"],  # This is the ONNX graph input name
        output_names=output_names_list,
        export_params=True,
        dynamo=True,
        dynamic_shapes=dynamic_input_shapes,  # Use the refined dynamic_input_shapes
    )
