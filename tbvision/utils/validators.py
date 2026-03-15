from fastapi import File, HTTPException, UploadFile


async def validate_pdf(file: UploadFile = File(...)) -> UploadFile:
    # Check extension
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a .pdf")

    # Check MIME type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid content type. Only PDFs allowed."
        )

    # Check file header
    content = await file.read(4)
    await file.seek(0)

    if content != b"%PDF":
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    return file
