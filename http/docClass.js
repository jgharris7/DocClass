
        // define the callAPI function that takes a first name and last name as parameters
        siteUrl="https://vy9a1rezhc.execute-api.us-east-1.amazonaws.com/Prod/docclass";
        const maxStringLength=7060;
        var returnval={
            prediction: "None",
            confidence: 0,
            model_name: "None"
        }
        var callAPI = (words)=>{
            setToWaiting();
            // instantiate a headers object
            var myHeaders = new Headers();
            // add content type header to object
            myHeaders.append("Content-Type", "application/json");
            // using built in JSON utility package turn object to string and store in a variable
            var requestOptions = {
                method: 'GET',
                headers: myHeaders,
                redirect: 'follow'
            };
            // make API call with parameters and use promises to get response
	       url=siteUrl;
           textLength=words.length;
	       url+='?'+("words="+encodeURIComponent(words.substring(1,maxStringLength)));
	       var data1=fetch(url,requestOptions)
              .then(response=>response.json())
              .then(data=> {console.log(data);
            this.returnval=data
            document.getElementById("prediction").innerHTML =data.prediction;
            document.getElementById("confidence").innerHTML=Number(data.confidence.substring(1,10)).toFixed(2);
            document.getElementById("model_name").innerHTML=data.model_name;
            document.getElementById("model_information").innerHTML=data.description;
            document.getElementById("text_length").innerHTML=textLength;
            document.getElementById("text_beginning").innerHTML=words.substring(1,42);
            document.getElementById("text_end").innerHTML=words.substring(textLength-41,textLength)
            if(textLength>maxStringLength) { 
                document.getElementById("text_length").innerHTML=textLength+" truncated to "+maxStringLength;
            }
        })
        .catch(error => {
            console.error('Error:', error)
            document.getElementById("model_information").innerHTML=error+" Could be text is too large (&gt 7600 characters)";
            document.getElementById("text_length").innerHTML=textLength;
        })
        ;
      
        };
   
        function clearit() {
           document.getElementById("prediction").innerHTML ="cleared ";
           document.getElementById("confidence").innerHTML=" cleared ";
           document.getElementById("model_name").innerHTML=" cleared ";
           console.log("cleared");
         };
         function setToWaiting() {
           document.getElementById("prediction").innerHTML =" waiting for response ";
           document.getElementById("confidence").innerHTML=" waiting for response ";
           document.getElementById("model_name").innerHTML=" waiting for response ";
           document.getElementById("model_information").innerHTML=" waiting for response ";
           console.log("cleared");
         };
