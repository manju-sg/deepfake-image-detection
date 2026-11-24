"""Build the detailed university project report as a PDF."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Deepfake_Detection_Project_Report.pdf"


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=draw_header_footer)])


def draw_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, height - 17 * mm, width - doc.rightMargin, height - 17 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(doc.leftMargin, height - 13 * mm, "Deep-Fake Detection Using Deep Learning")
    canvas.drawRightString(width - doc.rightMargin, 11 * mm, f"Page {doc.page}")
    canvas.drawString(doc.leftMargin, 11 * mm, "University project report | 24 November 2026")
    canvas.restoreState()


def make_styles():
    base = getSampleStyleSheet()
    navy = colors.HexColor("#16324F")
    teal = colors.HexColor("#007C91")
    slate = colors.HexColor("#334E68")
    light = colors.HexColor("#F0F4F8")
    styles = {
        "Title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26,
            leading=31, textColor=navy, alignment=TA_CENTER, spaceAfter=10,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12,
            leading=18, textColor=slate, alignment=TA_CENTER, spaceAfter=8,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17,
            leading=21, textColor=navy, spaceBefore=13, spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
            leading=16, textColor=teal, spaceBefore=9, spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4,
            leading=14, textColor=colors.HexColor("#243B53"), spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8,
            leading=11, textColor=colors.HexColor("#52606D"), spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13, leftIndent=12, firstLineIndent=-8, textColor=colors.HexColor("#243B53"), spaceAfter=3,
        ),
        "Code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=7.5,
            leading=10, textColor=colors.HexColor("#102A43"), backColor=light,
            borderColor=colors.HexColor("#D9E2EC"), borderWidth=0.5, borderPadding=7,
        ),
        "Callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.3,
            leading=14, textColor=navy, backColor=colors.HexColor("#E6FFFA"),
            borderColor=colors.HexColor("#83C5BE"), borderWidth=0.7, borderPadding=8,
            spaceBefore=5, spaceAfter=8,
        ),
    }
    return styles


def P(text, style):
    return Paragraph(text, style)


def bullet(text, style):
    return P(f"- {text}", style)


def table(data, widths, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F7FAFC")))
    t.setStyle(TableStyle(commands))
    return t


def architecture_diagram():
    d = Drawing(500, 105)
    boxes = [
        (8, 38, 104, 38, "Image upload", "JPG / PNG / WebP"),
        (132, 38, 112, 38, "Preprocessing", "Resize + normalize"),
        (264, 38, 112, 38, "ResNet-18", "Deep classifier"),
        (396, 38, 96, 38, "Decision", "Real / fake / unsure"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=colors.HexColor("#E6FFFA"), strokeColor=colors.HexColor("#007C91")))
        d.add(String(x + w / 2, y + 22, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#16324F")))
        d.add(String(x + w / 2, y + 10, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#52606D")))
    for x1, x2 in [(112, 132), (244, 264), (376, 396)]:
        d.add(Line(x1, 57, x2, 57, strokeColor=colors.HexColor("#52606D"), strokeWidth=1.2))
        d.add(Line(x2 - 5, 60, x2, 57, strokeColor=colors.HexColor("#52606D"), strokeWidth=1.2))
        d.add(Line(x2 - 5, 54, x2, 57, strokeColor=colors.HexColor("#52606D"), strokeWidth=1.2))
    d.add(String(250, 13, "Provenance side channel: EXIF / XMP / C2PA / provider hints", textAnchor="middle", fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#52606D")))
    return d


def build_report():
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=24 * mm, bottomMargin=19 * mm, title="Deep-Fake Detection Using Deep Learning",
        author="Manju SG",
    )
    story = []

    # Cover
    story += [Spacer(1, 22 * mm), P("Deep-Fake Detection", styles["Title"]), P("Using Deep Learning for Real vs AI-Generated Image Classification", styles["Subtitle"]), Spacer(1, 10 * mm)]
    story.append(HRFlowable(width="75%", thickness=2, color=colors.HexColor("#007C91"), hAlign="CENTER"))
    story += [Spacer(1, 10 * mm), P("University Project Report", styles["Subtitle"]), P("Prepared for implementation, demonstration, and future research extension", styles["Subtitle"]), Spacer(1, 15 * mm)]
    cover_data = [
        [P("Project focus", styles["Small"]), P("Binary classification of real and AI-generated images", styles["Body"])],
        [P("Deep-learning model", styles["Small"]), P("ImageNet-pretrained ResNet-18 fine-tuned for two classes", styles["Body"])],
        [P("Dataset", styles["Small"]), P("CIFAKE: real and Stable-Diffusion-generated images", styles["Body"])],
        [P("Application", styles["Small"]), P("Streamlit upload interface with visual and provenance analysis", styles["Body"])],
        [P("Date", styles["Small"]), P("24 November 2026", styles["Body"])],
    ]
    story.append(table(cover_data, [40 * mm, 115 * mm], header=False))
    story += [Spacer(1, 18 * mm), P("Important scope note: the delivered model is a research baseline. Its benchmark score is measured on CIFAKE-style images and must not be presented as guaranteed accuracy on arbitrary internet images.", styles["Callout"]), PageBreak()]

    story += [P("Executive Summary", styles["H1"]), P("This project implements an end-to-end deep-learning pipeline that accepts an image, preprocesses it, predicts the probability that it is AI-generated, and presents the result in a browser. The system uses transfer learning with ResNet-18, a convolutional neural network pretrained on ImageNet and fine-tuned for the two classes fake and real.", styles["Body"]), P("The project also adds a provenance side channel. It inspects surviving EXIF, XMP, PNG/JPEG text, and C2PA-related markers for hints associated with OpenAI/ChatGPT/DALL-E or Google/Gemini/Imagen. The provenance layer is intentionally separate from the visual model: metadata can be removed or forged, while a missing marker is not evidence that an image is real.", styles["Body"]), P("A complete runnable workflow was created: dataset download and conversion, training, validation, test evaluation, command-line prediction, and Streamlit upload UI. The saved demo checkpoint achieved 95.56% accuracy and 0.9926 ROC-AUC on a held-out CIFAKE-style test subset.", styles["Body"]), P("The principal engineering lesson is domain shift. CIFAKE images are very small and specialized, so the application now warns when high-resolution uploads are outside the training distribution and flags conflicts between camera metadata and the visual score.", styles["Callout"])]

    story += [P("1. Problem Definition and Objectives", styles["H1"]), P("AI image generators and face-manipulation systems can produce convincing media. A detector can help identify patterns associated with synthetic content, but it cannot establish truth in every case. The project therefore treats the task as a binary image-classification problem with an explicit uncertainty state.", styles["Body"]), P("Objectives", styles["H2"]), bullet("Build a reproducible deep-learning baseline for real vs AI-generated images.", styles["Bullet"]), bullet("Train and evaluate the model using a balanced dataset and held-out test split.", styles["Bullet"]), bullet("Provide an upload interface so a user can test an image without writing code.", styles["Bullet"]), bullet("Add provenance and metadata inspection for ChatGPT, DALL-E, Gemini, Imagen, and C2PA hints.", styles["Bullet"]), bullet("Document limitations, reproducibility commands, and a research path toward more general detectors.", styles["Bullet"])]

    story += [P("2. End-to-End System Architecture", styles["H1"]), architecture_diagram(), Spacer(1, 4 * mm), P("The main path is a supervised classifier. The image is converted to RGB, resized to 128 x 128 for the trained checkpoint, normalized using ImageNet statistics, and passed through ResNet-18. The final fully connected layer outputs two logits. A softmax converts these into class probabilities.", styles["Body"]), P("The side channel reads file-level evidence before or alongside classification. A provider marker can support an attribution hint, but it does not replace the learned visual prediction or prove that the entire image is unedited.", styles["Body"])]

    story += [P("3. Dataset and Preparation", styles["H1"]), P("The initial dataset was CIFAKE, a balanced collection of real CIFAR-10 images and AI-generated images produced with Stable Diffusion v1.4. The Hugging Face copy was downloaded and converted into the folder structure expected by torchvision.datasets.ImageFolder.", styles["Body"])]
    dataset_rows = [
        [P("Split", styles["Small"]), P("Full prepared data", styles["Small"]), P("Demo checkpoint subset", styles["Small"])],
        [P("Train", styles["Body"]), P("90,000", styles["Body"]), P("20,000", styles["Body"])],
        [P("Validation", styles["Body"]), P("10,000", styles["Body"]), P("4,000", styles["Body"])],
        [P("Test", styles["Body"]), P("20,000", styles["Body"]), P("8,000", styles["Body"])],
        [P("Classes", styles["Body"]), P("fake and real", styles["Body"]), P("fake and real", styles["Body"])],
    ]
    story.append(table(dataset_rows, [40 * mm, 55 * mm, 55 * mm]))
    story += [Spacer(1, 4 * mm), P("Folder layout", styles["H2"]), P("data/train/fake, data/train/real, data/val/fake, data/val/real, data/test/fake, and data/test/real. The project also contains data_quick for the balanced subset used to create the saved demo checkpoint.", styles["Body"]), P("The train/validation split was made with a fixed seed. For a research-grade experiment, images from the same source or generator should remain in one split only, and an unseen generator should be held out for cross-generator evaluation.", styles["Body"])]

    story += [P("4. Deep-Learning Method", styles["H1"]), P("The implemented model is ResNet-18 with ImageNet pretrained weights. Transfer learning was selected because a university-sized project usually has insufficient data and compute to train a general visual feature extractor from random initialization.", styles["Body"]), P("Model configuration", styles["H2"])]
    model_rows = [
        [P("Component", styles["Small"]), P("Implementation", styles["Small"])],
        [P("Backbone", styles["Body"]), P("torchvision.models.resnet18(weights=DEFAULT)", styles["Body"])],
        [P("Classifier", styles["Body"]), P("Linear layer with 2 output logits", styles["Body"])],
        [P("Input", styles["Body"]), P("RGB image resized to 128 x 128", styles["Body"])],
        [P("Augmentation", styles["Body"]), P("Horizontal flip, small rotation, brightness/contrast/saturation jitter", styles["Body"])],
        [P("Loss", styles["Body"]), P("Cross-entropy loss", styles["Body"])],
        [P("Optimizer", styles["Body"]), P("AdamW, learning rate 1e-4", styles["Body"])],
        [P("Hardware", styles["Body"]), P("NVIDIA GeForce RTX 3050 6 GB through Ubuntu WSL", styles["Body"])],
    ]
    story.append(table(model_rows, [42 * mm, 108 * mm]))
    story += [Spacer(1, 4 * mm), P("Training procedure", styles["H2"]), P("The best checkpoint is selected using validation ROC-AUC. After training, the saved checkpoint is reloaded and evaluated on the test directory. The code reports accuracy, precision, recall, F1-score, and ROC-AUC. A metric-label correction was made so ROC-AUC is computed consistently with the ImageFolder class mapping, where fake is class 0 and real is class 1.", styles["Body"])]

    story.append(KeepTogether([
        P("5. Results", styles["H1"]),
        P("The final demo checkpoint was trained for two epochs on the balanced data_quick subset. The metrics below come from the held-out test split and are included for reproducibility, not as a claim of universal real-world performance.", styles["Body"]),
    ]))
    result_rows = [
        [P("Metric", styles["Small"]), P("Validation", styles["Small"]), P("Test", styles["Small"])],
        [P("Accuracy", styles["Body"]), P("96.05%", styles["Body"]), P("95.56%", styles["Body"])],
        [P("ROC-AUC", styles["Body"]), P("0.9929", styles["Body"]), P("0.9926", styles["Body"])],
        [P("F1-score", styles["Body"]), P("0.9602", styles["Body"]), P("0.9551", styles["Body"])],
        [P("Precision", styles["Body"]), P("not retained in final log", styles["Body"]), P("0.9669", styles["Body"])],
        [P("Recall", styles["Body"]), P("not retained in final log", styles["Body"]), P("0.9435", styles["Body"])],
    ]
    story.append(table(result_rows, [47 * mm, 50 * mm, 53 * mm]))
    story += [Spacer(1, 4 * mm), P("Sanity checks", styles["H2"]), bullet("A known fake test image was classified as AI-generated with 100.00% AI probability.", styles["Bullet"]), bullet("A known real test image was classified as probably real with 0.31% AI probability.", styles["Bullet"]), bullet("The browser upload application returned HTTP status 200 from its local Streamlit server.", styles["Bullet"]), P("These checks confirm the pipeline is operational. They do not remove the need for broader evaluation on high-resolution and unseen-generator data.", styles["Body"])]

    story += [P("6. Upload Application", styles["H1"]), P("The Streamlit interface is implemented in app.py. It loads best_model.pt once, accepts JPG, JPEG, PNG, and WebP uploads, displays the image, runs the classifier, and shows AI-generated and real probabilities.", styles["Body"]), P("The final assessment layer has three important safeguards:", styles["H2"]), bullet("High-resolution warning: the initial checkpoint was trained on 32 x 32 CIFAKE content, so high-resolution uploads are marked outside the training domain.", styles["Bullet"]), bullet("Camera conflict warning: camera/gallery EXIF metadata can conflict with a high visual fake score; the app avoids presenting that conflict as proof.", styles["Bullet"]), bullet("Provider provenance hint: surviving provider markers are shown as hints and are not treated as a complete history of the file.", styles["Bullet"]), P("Launch command", styles["H2"]), P("streamlit run app.py --server.address 0.0.0.0 --server.port 8501", styles["Code"])]

    story += [P("7. Provenance and ChatGPT/Gemini Hints", styles["H1"]), P("The provenance scanner in provenance.py reads EXIF and other readable metadata and looks for C2PA-related markers and meaningful provider terms. It deliberately excludes ICC color profiles and JFIF fields from provider attribution because those fields can contain unrelated vendor text.", styles["Body"]), P("Supported interpretations", styles["H2"])]
    prov_rows = [
        [P("Signal", styles["Small"]), P("Meaning in this project", styles["Small"])],
        [P("OpenAI / ChatGPT / DALL-E term", styles["Body"]), P("Provider provenance hint; not proof that the image is unedited.", styles["Body"])],
        [P("Google / Gemini / Imagen term", styles["Body"]), P("Provider provenance hint; not proof that the image is unedited.", styles["Body"])],
        [P("C2PA / Content Credentials marker", styles["Body"]), P("A provenance container may be present; issuer still needs validation.", styles["Body"])],
        [P("Camera EXIF metadata", styles["Body"]), P("Supports camera-captured origin, but metadata can be changed.", styles["Body"])],
        [P("No marker", styles["Body"]), P("No readable evidence found; never interpret this as proof of human creation.", styles["Body"])],
    ]
    story.append(table(prov_rows, [60 * mm, 90 * mm]))
    story += [Spacer(1, 4 * mm), P("For supported OpenAI files, the original file can be checked with OpenAI Verify. Google SynthID is a vendor watermark technology for supported generated content. A local text scan cannot independently decode a vendor watermark, so the app presents these checks as hints rather than definitive attribution.", styles["Body"])]

    story += [P("8. Reproducibility", styles["H1"]), P("The project was run in Ubuntu WSL because Windows Smart App Control blocked PyTorch native DLLs. WSL exposed the RTX 3050 GPU to Linux, where the model trained successfully.", styles["Body"]), P("Environment setup", styles["H2"]), P("python3 -m venv /home/$USER/deepfake-venv\n/home/$USER/deepfake-venv/bin/pip install -r requirements.txt", styles["Code"]), P("Dataset preparation", styles["H2"]), P("python prepare_cifake.py\npython prepare_cifake.py --limit-per-class 2000", styles["Code"]), P("Training", styles["H2"]), P("python train.py --data-dir data_quick --epochs 2 --batch-size 128 --image-size 128 --workers 4", styles["Code"]), P("Command-line prediction", styles["H2"]), P("python predict.py path/to/image.jpg", styles["Code"]), P("Project structure", styles["H2"]), P("train.py - model training and evaluation\npredict.py - command-line prediction\napp.py - Streamlit upload interface\nprepare_cifake.py - dataset download and conversion\nprovenance.py - metadata and provider-hint scanner\nbest_model.pt - saved trained checkpoint\nPAPERS.md - research paper list and implementation path", styles["Code"])]

    story += [P("9. Limitations, Safety, and Responsible Use", styles["H1"]), bullet("CIFAKE is a useful baseline but is not a complete representation of current image generators. Its images are low resolution and its fake distribution is tied to a specific generation setup.", styles["Bullet"]), bullet("A high score is a model belief, not proof. False positives can wrongly accuse real photographers, and false negatives can allow synthetic media to pass.", styles["Bullet"]), bullet("Metadata can be removed, edited, or lost through screenshots and platform recompression.", styles["Bullet"]), bullet("Provider attribution is only defensible when a trusted provenance signal survives and validates. A text string alone is not a cryptographic proof.", styles["Bullet"]), bullet("The system should not be used alone for legal, employment, academic misconduct, or public accusations.", styles["Bullet"]), P("The app's uncertainty and conflict states are intentional safety features. A future version should use calibration, out-of-distribution detection, and evaluation against unseen generators before making stronger claims.", styles["Callout"])]

    story += [P("10. Future Work", styles["H1"]), bullet("Retrain using high-resolution, multi-generator data such as GenImage and evaluate on a generator held out from training.", styles["Bullet"]), bullet("Add a separate face-manipulation track using FaceForensics++ for face swaps and manipulated videos.", styles["Bullet"]), bullet("Add temperature scaling or another calibration method so probabilities better reflect observed correctness.", styles["Bullet"]), bullet("Add a frequency-domain branch and compare spatial, frequency, and fused representations.", styles["Bullet"]), bullet("Add Grad-CAM or similar explanations so the user can see which regions influenced the visual score.", styles["Bullet"]), bullet("Add automated C2PA validation through a trusted verifier when a suitable service or SDK is available.", styles["Bullet"]), bullet("Create a larger real-world test set containing phone photos, screenshots, social-media recompressions, and images from recent generators.", styles["Bullet"])]

    story += [P("References", styles["H1"]), P("1. Bird, J. J. and Lotfi, A. A. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. https://arxiv.org/abs/2303.14126", styles["Small"]), P("2. Zhu, M. et al. GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. https://arxiv.org/abs/2306.08571", styles["Small"]), P("3. Rossler, A. et al. FaceForensics++: Learning to Detect Manipulated Facial Images. https://arxiv.org/abs/1901.08971", styles["Small"]), P("4. Wang, S.-Y. et al. CNN-generated images are surprisingly easy to spot... for now. https://arxiv.org/abs/1912.11035", styles["Small"]), P("5. PyTorch. Transfer Learning for Computer Vision Tutorial. https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html", styles["Small"]), P("6. OpenAI. Content provenance and verification. https://developers.openai.com/api/docs/guides/content-provenance", styles["Small"]), P("7. Google DeepMind. SynthID. https://deepmind.google/models/synthid/", styles["Small"])]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()

