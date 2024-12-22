const multerUpload = require('./multer').multerUpload;
const router = require('express').Router();
router.post('/upload', multerUpload);
module.exports = router;