from tbvision.services.classifier import ClassifierService
from tbvision.utils.check_internet import check_internet_connection


class Analyzer:
    def __init__(self, image, classifier_service, metadata):
        self.image = image
        self.metadata = metadata
        self.classifier_service: ClassifierService = classifier_service
        self.offline_mode = not check_internet_connection()

    def analyze(self):
        gradcam_data = {}
        gradcam_image = None
        try:
            pred_data = self.classifier_service.predict(self.image)
            gradcam_data = self.classifier_service.analyze_gradcam(pred_data)
            try:
                gradcam_image = self.classifier_service.create_gradcam_overlay(
                    self.image, gradcam_data["heatmap"]
                )
            except Exception as e:
                print(f"Grad-CAM++ overlay generation failed: {e}")
        except Exception as err:
            print(f"Error occured while analyzing: {err}")
            raise

        if self.offline_mode:
            print("Generating offline explanation...")
            explanation = {}  # TODO
            pred_data["mode"] = "offline"
            pred_data["evidence"] = []
        else:
            print("Generating online explanation...")
            pred_data["mode"] = "online"
            pred_data["evidence"] = []
            explanation = {}  # TODO

        pred_data["gradcam_region"] = gradcam_data.get("description")
        pred_data["gradcam_image"] = gradcam_image
        pred_data["explanation"] = explanation

        pred_data.pop("image_tensor", None)
        return pred_data
