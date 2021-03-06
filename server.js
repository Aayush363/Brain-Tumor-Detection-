const Express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const ejs = require("ejs");


var prediction = "", img_path = "";

var app = Express();
app.use(bodyParser.urlencoded({extended: true}));
app.use(Express.static(__dirname + '/public'));
app.set('view engine', 'ejs');


app.get("/", function(req, res) {
    res.render("main", {img: img_path, pred: prediction});
});

var Storage = multer.diskStorage({
    destination: function(req, file, callback) {
        callback(null, "./public/uploaded_images");
    },
    filename: function(req, file, callback) {
        callback(null,file.originalname);
    }
});

var upload = multer({
    storage: Storage
});

app.post("/", upload.single('myFile'), function(req, res) {
    console.log(req.file.originalname);
    var spawn = require("child_process").spawn;
    var process = spawn('python', ["./script.py", req.file.originalname]);
    process.stdout.on('data', function(data) {
        img_path = "./uploaded_images/" + req.file.originalname;
        console.log(data.toString());
        if(data.toString() == 1)
            prediction = "Malignant";
        else
            prediction = "Benign";
        res.redirect("/");
    })
}); 

app.listen(3000, function(a) {
    console.log("Listening to port 3000");
});