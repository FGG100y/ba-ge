
# from utf8mb4 (still technically Unicode, but with notoriously terrible support for
# most Unicode characters) to utf8mb4_unicode_ci (say for chinese chars):
mydb_name = "scraping"  # database 'scraping'
mytbl_name = "pages"
tbl_field1 = "title"
tbl_field2 = "content"
extend_unicode_support = f"""
ALTER DATABASE {mydb_name} CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
ALTER TABLE {mytbl_name} CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE {mytbl_name} CHANGE {tbl_field1} {tbl_field1} VARCHAR(200) CHARACTER SET utf8mb4 COLLATE
utf8mb4_unicode_ci;
ALTER TABLE {mytbl_name} CHANGE {tbl_field2} {tbl_field2} VARCHAR(10000) CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
"""
