const multer = require("multer");
const path = require("path");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, '../../data'))
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname)
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10000000 }, // 10MB file size limit
});

module.exports.multerUpload = async (request, response) => {
  upload.single('file')(request, response, (err) => {
    if (err) {
      console.error(err);
      return response.status(500).send('Error uploading file');
    }

    console.log('Request body:', request.body);
    
    if (!request.file) {
      return response.status(400).send('No file uploaded');
    }

    console.log('File details:', request.file);
    console.log('File uploaded successfully');
    let relativePath = "/data/" + request.file.filename;
    response.status(200).json({
      message: 'File uploaded successfully',
      filePath: relativePath,
      body: request.body
    });
  });
};