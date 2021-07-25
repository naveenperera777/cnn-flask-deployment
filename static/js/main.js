$(document).ready(function () {
  // Init
  const submit = document.getElementById("homebtn");
  const age = document.getElementById("age");
  const xrayImage = document.getElementById("xrayImage");

  console.log("age", age.value, xrayImage.value);

  submit.disabled = true;
  function handleClick() {
    console.log("handleClick");
  }

  function handleChange(e) {
    console.log("handleChange", e.target.value, xrayImage.value);
    if (
      e.target.value > 0 &&
      e.target.value < 101 &&
      e.target.value.length != 0
    ) {
      submit.disabled = false;
    } else {
      submit.disabled = true;
    }
  }

  function uploadEvent(e) {
    console.log("upload", e.target.value.length);
    if (e.target.value.length == 0) {
      submit.disabled = true;
    } else if (age.value > 0 && age.value < 101) {
      submit.disabled = false;
    }
  }

  submit.addEventListener("click", handleClick);
  age.addEventListener("input", handleChange);
  xrayImage.addEventListener("change", uploadEvent);
});
