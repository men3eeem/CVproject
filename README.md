# Egyptian Vehicle & License Plate Detection with OCR

An end-to-end computer vision pipeline tailored for Egyptian traffic data. The system sequentially detects vehicles, localizes license plates using a dedicated secondary model, and extracts the text using Optical Character Recognition (OCR).

## Pipeline Architecture
1. **Vehicle Detection:** Identifies and localizes cars within the input frame or video.
2. **License Plate Detection:** Crops and isolates the license plate region from the detected vehicles using a specialized model.
3. **OCR (Optical Character Recognition):** Converts the extracted license plate image into readable text strings.

## Dataset
This project is built and evaluated using Egyptian traffic and license plate data:
* [EALPR Dataset](https://github.com/ahmedramadan96/EALPR)

## Project Team
* Abdelmonem Gad
* Abdelrhman Elyamny
* Aya Yaser
* Yousef Hagrs
* Menna Emad
