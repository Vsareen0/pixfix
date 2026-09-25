-- PixFix database schema (PostgreSQL). Database: db_pixfix
-- Generated from app/models.py; the app also creates these tables automatically on startup.
--   createdb db_pixfix && psql db_pixfix -f schema.sql

CREATE TYPE image_status AS ENUM ('pending', 'processing', 'done', 'failed');

CREATE TABLE tbl_users (
	user_id SERIAL NOT NULL, 
	username VARCHAR(50) NOT NULL, 
	email VARCHAR(100) NOT NULL, 
	password VARCHAR(255) NOT NULL, 
	is_admin BOOLEAN DEFAULT 'false' NOT NULL, 
	is_active BOOLEAN DEFAULT 'true' NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	last_login_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (user_id)
);
CREATE UNIQUE INDEX ix_tbl_users_email ON tbl_users (email);

CREATE TABLE tbl_images (
	img_id SERIAL NOT NULL, 
	user_id INTEGER NOT NULL, 
	original_filename VARCHAR(255), 
	original_path VARCHAR(255) NOT NULL, 
	mask_path VARCHAR(255), 
	result_path VARCHAR(255), 
	status image_status NOT NULL, 
	model_used VARCHAR(20), 
	width INTEGER, 
	height INTEGER, 
	file_size BIGINT, 
	processing_time FLOAT, 
	error_message TEXT, 
	upload_time TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	completed_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (img_id), 
	FOREIGN KEY(user_id) REFERENCES tbl_users (user_id) ON DELETE CASCADE
);
CREATE INDEX ix_tbl_images_upload_time ON tbl_images (upload_time);
CREATE INDEX ix_tbl_images_status ON tbl_images (status);
CREATE INDEX ix_tbl_images_user_id ON tbl_images (user_id);
