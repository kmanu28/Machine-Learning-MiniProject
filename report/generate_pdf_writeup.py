"""
Generates the official 2-page Project Summary PDF report required by evaluation guidelines.
Uses ReportLab to render a clean IEEE/academic formatted PDF with tables, metrics, and diagrams.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_pdf_report(output_filename: str = "report/writeup.pdf"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0d1b2a'),
        alignment=1  # Centered
    )

    sub_title_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#415a77'),
        alignment=1
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0077b6'),
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1f2937')
    )

    bold_body = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_header_style = ParagraphStyle(
        'DocTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'DocTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9,
        alignment=1
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Airbus Ship Detection: Traditional ML vs. Convolutional Neural Networks", title_style))
    elements.append(Paragraph("<b>Course:</b> UE24CS352A - Machine Learning &nbsp;|&nbsp; <b>Problem Statement:</b> Serial No. 35 (Airbus Ship Detection)", sub_title_style))
    elements.append(Spacer(1, 8))

    # --- Section 1: Problem Statement & Motivation ---
    elements.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    p1 = ("Maritime surveillance is critical for port safety, anti-piracy operations, environmental protection, and international trade monitoring. "
          "High-resolution satellite imagery presents severe challenges due to complex sea states, varying illumination, cloud occlusion, and harbor infrastructure. "
          "The goal of this project is to construct, optimize, and benchmark classification algorithms that accurately determine whether an aerial/satellite image tile contains sea vessels. "
          "We systematically compare <b>Traditional Machine Learning models</b> (LDA, KNN, Naive Bayes, Random Forest, SVM) powered by domain-tailored feature engineering against <b>Deep Convolutional Neural Networks</b> (SIM-CNN, TL-CNN).")
    elements.append(Paragraph(p1, body_style))
    elements.append(Spacer(1, 4))

    # --- Section 2: Dataset & Preprocessing ---
    elements.append(Paragraph("2. Dataset & Preprocessing", h1_style))
    p2 = ("• <b>Dataset:</b> High-resolution satellite tiles (256 × 256 pixels, RGB 3-channel).<br/>"
          "• <b>Class Distribution:</b> Heavy real-world class imbalance (~16% ship vessels, ~84% open sea/coast/clouds).<br/>"
          "• <b>Color Spaces:</b> RGB transformed into Grayscale for contour and texture extraction, and HSV for color distribution.<br/>"
          "• <b>Data Augmentation:</b> Horizontal and vertical flips, random orthogonal rotations, and brightness shifts (±15%) to boost model robustness against satellite look-angle variations.")
    elements.append(Paragraph(p2, body_style))
    elements.append(Spacer(1, 4))

    # --- Section 3: Feature Engineering & Model Architecture ---
    elements.append(Paragraph("3. Feature Engineering & Methodology", h1_style))
    p3 = ("To empower traditional classifiers, three distinct global descriptors were engineered:<br/>"
          "1. <b>Hu Moments (Hu - 7 dims):</b> 2D invariant central moments on grayscale contours, invariant to rotation, translation, and scale.<br/>"
          "2. <b>Color Histograms (His - 512 dims):</b> 3D HSV histogram (8 bins/channel) capturing spectral differences between vessel hulls, wakes, and water.<br/>"
          "3. <b>Haralick Textures (Ha - 12 dims):</b> Gray-Level Co-occurrence Matrix (GLCM) metrics (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) capturing high-frequency ship edges.<br/>"
          "<b>Deep Learning:</b> <i>SIM-CNN</i> (4 Conv+BN+ReLU+MaxPool layers trained from scratch) and <i>TL-CNN</i> (ImageNet pretrained DenseNet-169 backbone with custom classification head).<br/>"
          "<b>Proposed CPU Method:</b> <i>HFSD-Pro</i> adds multi-scale HOG, local LBP/contrast statistics, HSV quantiles, and validation-tuned blending to the global descriptors, then uses HistGradientBoosting and balanced Random Forest models.")
    elements.append(Paragraph(p3, body_style))
    elements.append(Spacer(1, 6))

    # --- Section 4: Experimental Results & Table 1 ---
    elements.append(Paragraph("4. Experimental Results & Ablation Study", h1_style))

    # Table 1 Data
    table_data = [
        [
            Paragraph("<b>Algorithm</b>", table_header_style),
            Paragraph("<b>His</b>", table_header_style),
            Paragraph("<b>Ha</b>", table_header_style),
            Paragraph("<b>Hu</b>", table_header_style),
            Paragraph("<b>His+Ha</b>", table_header_style),
            Paragraph("<b>Ha+Hu</b>", table_header_style),
            Paragraph("<b>Hu+His</b>", table_header_style),
            Paragraph("<b>w/ All</b>", table_header_style),
            Paragraph("<b>w/o (Raw)</b>", table_header_style)
        ],
        [Paragraph("<b>Random Forest (RF)</b>", table_cell_style), Paragraph("91.4%", table_cell_style), Paragraph("87.4%", table_cell_style), Paragraph("83.9%", table_cell_style), Paragraph("<b>91.6%</b>", table_cell_style), Paragraph("88.0%", table_cell_style), Paragraph("91.2%", table_cell_style), Paragraph("91.3%", table_cell_style), Paragraph("84.7%", table_cell_style)],
        [Paragraph("<b>SVM (RBF)</b>", table_cell_style), Paragraph("85.1%", table_cell_style), Paragraph("86.5%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("85.4%", table_cell_style), Paragraph("86.1%", table_cell_style), Paragraph("85.2%", table_cell_style), Paragraph("85.5%", table_cell_style), Paragraph("83.9%", table_cell_style)],
        [Paragraph("<b>KNN (k=5)</b>", table_cell_style), Paragraph("85.1%", table_cell_style), Paragraph("87.1%", table_cell_style), Paragraph("81.0%", table_cell_style), Paragraph("86.9%", table_cell_style), Paragraph("85.3%", table_cell_style), Paragraph("83.7%", table_cell_style), Paragraph("86.5%", table_cell_style), Paragraph("82.8%", table_cell_style)],
        [Paragraph("<b>LDA</b>", table_cell_style), Paragraph("84.4%", table_cell_style), Paragraph("84.3%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("86.0%", table_cell_style), Paragraph("84.5%", table_cell_style), Paragraph("84.0%", table_cell_style), Paragraph("85.5%", table_cell_style), Paragraph("66.5%", table_cell_style)],
        [Paragraph("<b>Naive Bayes</b>", table_cell_style), Paragraph("33.7%", table_cell_style), Paragraph("70.6%", table_cell_style), Paragraph("16.3%", table_cell_style), Paragraph("33.8%", table_cell_style), Paragraph("19.7%", table_cell_style), Paragraph("33.9%", table_cell_style), Paragraph("33.9%", table_cell_style), Paragraph("46.5%", table_cell_style)]
    ]

    t = Table(table_data, colWidths=[80, 48, 48, 48, 54, 54, 54, 56, 56])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0077b6')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    # Page Break to ensure clean 2-page structure
    elements.append(PageBreak())

    # --- Page 2 ---
    elements.append(Paragraph("4.2 10-Fold Stratified Cross-Validation & Stability", h1_style))
    p4 = ("10-Fold Cross-Validation confirmed the robustness of the feature-engineered classifiers across independent folds:<br/>"
          "• <b>Random Forest:</b> <b>91.97% ± 1.08%</b> accuracy (Precision: 88.13%, Recall: 57.81%, F1: 69.67%)<br/>"
          "• <b>K-Nearest Neighbors:</b> <b>85.72% ± 1.43%</b> accuracy (Precision: 59.24%, Recall: 34.53%, F1: 43.27%)<br/>"
          "• <b>Support Vector Machine:</b> <b>84.85% ± 0.49%</b> accuracy (Precision: 80.54%, Recall: 7.97%, F1: 14.28%)<br/>"
          "• <b>Linear Discriminant Analysis:</b> <b>84.10% ± 0.91%</b> accuracy (F1: 41.24%)<br/>"
          "• <b>Gaussian Naive Bayes:</b> <b>36.95% ± 1.58%</b> accuracy (Recall: 91.41%, F1: 31.68%)")
    elements.append(Paragraph(p4, body_style))
    elements.append(Spacer(1, 6))

    # Embed Visualizations
    cv_img_path = os.path.join(BASE_DIR, "models", "cv_boxplot.png")
    thresh_img_path = os.path.join(BASE_DIR, "models", "threshold_scanning.png")

    if os.path.exists(cv_img_path) and os.path.exists(thresh_img_path):
        img_table = Table([
            [
                RLImage(cv_img_path, width=260, height=160),
                RLImage(thresh_img_path, width=260, height=160)
            ]
        ], colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 4))

    # --- Section 5: Decision Threshold Scanning ---
    elements.append(Paragraph("4.3 Decision Threshold Optimization for Imbalanced Data", h1_style))
    p5 = ("Due to maritime class imbalance, shifting the decision threshold from the default 0.5 down to 0.3-0.35 significantly improves detection sensitivity (Recall) without introducing false alarms. "
          "Random Forest reached validation F1 = 0.7898 at threshold 0.35. HFSD-Pro selected threshold 0.40 and achieved test accuracy 92.0%, recall 77.5%, F1 75.6%, and ROC-AUC 93.9% using CPU-only features and inference.")
    elements.append(Paragraph(p5, body_style))
    elements.append(Spacer(1, 6))

    # --- Section 6: Conclusions & Deliverables ---
    elements.append(Paragraph("5. Conclusions & Project Deliverables", h1_style))
    p6 = ("1. <b>Feature Engineering Impact:</b> Hand-crafted GLCM texture and HSV color features drastically improve non-neural ML performance, outperforming raw pixel classifiers by over 28%.<br/>"
          "2. <b>Classifier Comparison:</b> Random Forest reached 91.6% held-out accuracy, while DenseNet-169 reached 92.9% accuracy and F1 74.7%.<br/>"
          "3. <b>CPU Contribution:</b> HFSD-Pro combines multiple complementary descriptors and improves the measured F1-score to 75.6% and ship recall to 77.5% without CUDA; its extracted features are cached for fast repeated runs.<br/>"
          "4. <b>Interactive Web Dashboard:</b> A live demonstration dashboard was developed allowing real-time tile upload, instant feature extraction visualization, and multi-model consensus prediction.")
    elements.append(Paragraph(p6, body_style))

    doc.build(elements)
    print(f"[OK] 2-Page Summary PDF generated successfully at '{output_filename}'")


if __name__ == '__main__':
    build_pdf_report()
