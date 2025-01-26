Students choose own topics from the attached list of projects and work in groups of 2 - 3 students. The deadline for submission is February 03, 2025. The submission must include: a video, full presentation slides and source code.

After completing the project, each group will need to record their presentation, upload it to YouTube, and submit the presentation video link along with links to the slides and source code. In the presentation, please clarify the following:

The reason for choosing this project.
Practical applications of the project in real life.
Explanation of the technologies used in the project.
The entire process of the software’s operation.
Any additional information.
Software that includes a complete, convenient, and user-friendly interface will receive higher scores compared to programs that run only in the terminal. Additionally, if you have more creative ideas for instance, instead of simply using facial recognition, you could develop software for attendance tracking or ticket verification... you will also receive higher marks.

The teacher will evaluate your work based on: video and  slides (for midterm mark), and source code (for final mark). Once grading is complete, the department will announce the scores to all students. Wishing you the best of success!
Please also include in your slide the list of group members with complete student ID numbers.

**Đề bài**
 Email Spam Classification Using Logistic Regression 
• Objective: Use a logistic regression model to classify emails into two categories: spam 
and not spam. 
• Dataset: Use a spam email dataset, such as the Spam Email Dataset from the UCI 
Machine Learning Repository. 
• Implementation: Students work with simple text data processing and apply logistic 
regression for classification. 

--------------------------------------------------------------------------------
Cải tiến
1. Giao diện Người Dùng Hoàn Thiện Hơn
Prompt:

Thêm tính năng tải lên file CSV:
"Hãy thêm một tính năng trong ứng dụng cho phép người dùng tải lên tệp CSV chứa danh sách email. Sau đó, ứng dụng sẽ phân loại tất cả các email trong file và hiển thị kết quả trong một bảng. Bảng cần có các cột: Email Content, Prediction (Spam/Not Spam), Spam Probability. Ngoài ra, cung cấp nút để tải xuống file kết quả dưới dạng CSV."

Hiển thị biểu đồ:
"Thêm một biểu đồ tròn hoặc biểu đồ cột hiển thị tỷ lệ phần trăm email được phân loại là spam và không phải spam. Ngoài ra, tạo một biểu đồ Word Cloud để hiển thị những từ phổ biến nhất trong các email spam và không spam."

2. Cải Tiến Mô Hình
Prompt:

Tối ưu hóa Logistic Regression:
"Thử nghiệm tối ưu hóa Logistic Regression bằng cách điều chỉnh tham số C (điều chỉnh mức độ phạt) và solver (thuật toán tối ưu hóa). Sử dụng Cross-Validation để đánh giá mô hình với các tập dữ liệu khác nhau và chọn mô hình có độ chính xác cao nhất."

Lọc và chuẩn hóa dữ liệu:
"Thêm bước tiền xử lý dữ liệu để chuẩn hóa văn bản. Cụ thể: loại bỏ khoảng trắng thừa, ký tự đặc biệt, số, và thực hiện lemmatization để chuẩn hóa từ vựng. Bước này nên thực hiện trước khi vector hóa bằng TF-IDF."

3. Hiển Thị Chi Tiết Giải Thích
Prompt:

Giải thích kết quả dự đoán:
"Sau khi phân loại một email, hãy hiển thị danh sách các từ quan trọng nhất trong email đã ảnh hưởng đến quyết định của mô hình (dựa trên trọng số TF-IDF). Giải thích cụ thể tại sao email được phân loại là spam hoặc không spam dựa trên các từ khóa này."
4. Tích Hợp Kiểm Tra Thực Tế
Prompt:

Tích hợp gửi email thử nghiệm:
"Thêm tính năng cho phép người dùng nhập nội dung email và một địa chỉ email để gửi thử nghiệm. Khi email được gửi, ứng dụng sẽ phân loại nội dung và hiển thị kết quả (spam hoặc không spam). Ngoài ra, thông báo trạng thái gửi email thành công hoặc thất bại."
5. Lưu Trữ Kết Quả
Prompt:

Tạo lịch sử phân loại:
"Lưu lại tất cả kết quả phân loại của các email trong một file CSV . Hiển thị một bảng lịch sử trên ứng dụng, bao gồm nội dung email, dự đoán (spam hoặc không spam), xác suất spam, và thời gian phân loại. Cho phép người dùng tải xuống lịch sử này dưới dạng file CSV."
6. Báo Cáo Hiệu Suất
Prompt:

Hiển thị các chỉ số đánh giá:
"Sau khi huấn luyện mô hình, hãy hiển thị các chỉ số đánh giá hiệu suất như precision, recall, F1-score, và accuracy. Đồng thời, vẽ ma trận nhầm lẫn (confusion matrix) để người dùng hiểu rõ hơn về các trường hợp dự đoán đúng và sai của mô hình."