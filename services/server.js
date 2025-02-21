require("dotenv").config({ path: "../api.env.local" }); 
const express = require("express");
const cors = require("cors");
var bodyParser = require('body-parser'); 
const OpenAIApi = require('openai');


const app = express();
const port = 5000;

// Middleware
app.use(bodyParser.json({limit: '50mb', type: 'application/json'}));
app.use(cors());
app.use(express.json());


const openai = new OpenAIApi({
  apiKey: process.env.OPENAI_API_KEY
});

example_json = {
  "columns": [
        "id",
        "first_name",
        "last_name",
        "email",
        "gender",
        "ip_address",
        "birthdate",
        "country",
        "salary",
        "job_title"
    ],
    "description": [
        {"id": "A unique identifier for each record in the dataset."},
        {"first_name": "The given name of the individual."}]
       
    };

// Route to analyze data
app.post("/analyze", async (req, res) => {
  try {
    const { prompt } = req.body;

    // if(window.localStorage.getItem('ColumnAnalysis') !== null){
    //   res.json({ result: window.localStorage.getItem('ColumnAnalysis') });
    //   return;
    // }

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      store: true,
      store: true,
      temperature: 0.7,
      response_format:{ "type": "json_object" },   
      messages: [
        
        {"role": "system", "content": JSON.stringify(prompt),
        "role": "user", "content": "Use the JSON data and give me a JSON output with all columns and their description. From the data analyse the column description. Output JSON should have two properties, the columns form the data and their description",
        "role": "user", "content": "The output JSON data schema should be like this. " + JSON.stringify(example_json)
        }
      ]
    });

    //var filteredContent = response.choices[0].message.replace(/'/g, '"');
    //localStorage.setItem('ColumnAnalysis', response.choices[0].message);
    res.json({ result: response.choices[0].message });

  } catch (error) {
    console.error("Error during OpenAI request:", error);
    res.status(500).json({ error: "An error occurred while processing your request." });
  }
});





// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
