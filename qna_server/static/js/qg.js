    

$(document).ready(function() {
  $("#QA_generation_button").click(function(){
    $("#QA_generation_button").attr("disabled", true)
    $("#QA_generation_button").css("cursor", "default").css('opacity',0.5)
    $("#QA_pairs").hide()
    $("#QA_header").text("Generating Q&A...")
    $("#QA_div").show()
    console.log($("#QA_input").val())

    //Generate questions
    var postRequest = $.post("/qg_with_qa", {"input_text": $("#QA_input").val()});
    postRequest.done(function (output) {
        console.log(output)
        //output = JSON.parse(output)
        questions = output['questions']
        answers = output['answers']
        var qa_display_text = "";
        for (var i = 0; i < questions.length; i++) {
          qa_display_text += "Q: " + questions[i] + "\n" + "A: " + answers[i] + "\n\n";
        }
        $("#QA_header").text("Q&A")
        $("#QA_pairs").text(qa_display_text)
        $("#QA_header").show()
        $("#QA_pairs").show()
        $("#QA_generation_button").attr("disabled", false).css("cursor", "pointer").css('opacity',1.0)
    });

    postRequest.fail(function (error) {
        console.log(error)
        $("#QA_pairs").text("Error: question generation failed: " + error)
    });

  });
});