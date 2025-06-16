@testset "File Management" begin
    # test image, audio, text, video, pdf
    test_files = [
        "input/example.jpg",
        "input/example.m4a",
        "input/example.txt",
        "input/example.mp4",
        "input/example.pdf",
    ]
    uploaded_files = []
    for test_file in test_files
        # 1) Upload the file.
        upload_result = upload_file(secret_key, test_file; display_name="Test Upload")
        push!(uploaded_files, upload_result)
        @test haskey(upload_result, "name")
        file_name = upload_result["name"]

        # 2) Retrieve file metadata.
        get_result = get_file(secret_key, file_name)
        @test get_result["name"] == file_name
    end

    # Generate content with all files
    @info "Sleeping for 5 seconds to allow files to be processed"
    sleep(5) # allow time for the files to be processed 
    contents = ["What files do you have?", uploaded_files...]
    model = "gemini-2.0-flash-lite"
    response = generate_content(secret_key, model, contents;)

    # Test that all the files are available, then delete them
    list_result = list_files(secret_key; page_size=10)
    for file in uploaded_files
        @test any(f -> f["name"] == file["name"], list_result)
        delete_status = delete_file(secret_key, file["name"])
        @test delete_status == 200 || delete_status == 204
    end
end
