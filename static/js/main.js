document.addEventListener("DOMContentLoaded", () => {
    // 1. Mobile Menu Navigation Toggle
    const menuToggle = document.getElementById("menu-toggle");
    const navLinks = document.getElementById("nav-links");
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", () => {
            navLinks.classList.toggle("show");
            // Toggle hamburger menu to cross icon style animation if needed
            menuToggle.classList.toggle("open");
        });
    }

    // 2. Form Submission Loading Spinner State
    const form = document.getElementById("prediction-form");
    const btnSubmit = document.getElementById("btn-submit");
    const spinner = document.getElementById("spinner");
    
    if (form && btnSubmit && spinner) {
        form.addEventListener("submit", (event) => {
            // Check HTML5 validations
            if (form.checkValidity()) {
                // Show loading spinner
                spinner.classList.remove("hidden");
                // Update text inside button
                const btnText = btnSubmit.querySelector(".btn-text");
                if (btnText) {
                    btnText.textContent = "Analyzing Soil & Climate...";
                }
                // Disable button asynchronously to allow submission to proceed
                setTimeout(() => {
                    btnSubmit.disabled = true;
                }, 10);
            }
        });
    }
});
