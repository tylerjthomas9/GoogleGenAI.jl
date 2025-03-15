using GoogleGenAI
secret_key = ENV["GOOGLE_API_KEY"]

test_files = [
    "test/input/example.jpg",
    "test/input/example.m4a",
    "test/input/example.txt",
    "test/input/example.mp4",
    "test/input/example.pdf",
]
uploaded_files = []
for test_file in test_files
    # 1) Upload the file.
    upload_result = upload_file(secret_key, test_file; display_name="Test Upload")
    push!(uploaded_files, upload_result)
    file_name = upload_result["name"]

    # 2) Retrieve file metadata.
    get_result = get_file(secret_key, file_name)
end

contents = ["What files do you have?", uploaded_files...]
model = "gemini-2.0-flash-lite"
response = generate_content(secret_key, model, contents;)
