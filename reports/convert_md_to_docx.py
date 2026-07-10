import pypandoc

print("Downloading pandoc...")
pypandoc.download_pandoc()

print("Converting file...")
input_file = r"d:\Eye\Eye-diseases-classification\reports\Complete_Scientific_Manuscript.md"
output_file = r"d:\Eye\Eye-diseases-classification\reports\Complete_Scientific_Manuscript.docx"

try:
    pypandoc.convert_file(input_file, 'docx', outputfile=output_file, extra_args=['--resource-path=C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e'])
    print(f"Conversion successful: {output_file}")
except Exception as e:
    print(f"Error during conversion: {e}")
