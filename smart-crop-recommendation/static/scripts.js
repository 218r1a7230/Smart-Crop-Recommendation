document.addEventListener("DOMContentLoaded", function () {
    const graphContainer = document.getElementById("graphContainer");
    const graphImage = document.getElementById("graphImage");
    const prevButton = document.getElementById("prevGraph");
    const nextButton = document.getElementById("nextGraph");
    const closeButton = document.getElementById("closeGraphSection");
    const graphSection = document.getElementById("graphSection");

    // ✅ Ensure all elements exist before proceeding
    if (!graphContainer || !graphImage || !prevButton || !nextButton || !closeButton || !graphSection) {
        console.warn("Graph container or navigation buttons not found.");
        return;
    }

    // ✅ Parse graph data safely
    let graphs = [];
    try {
        let graphData = graphContainer.getAttribute("data-graphs");
        if (graphData) {
            graphs = JSON.parse(graphData);
        }
    } catch (error) {
        console.error("Error parsing graph data:", error);
    }

    // ✅ Ensure graphs exist before proceeding
    if (!Array.isArray(graphs) || graphs.length === 0) {
        console.warn("No graphs found!");
        graphImage.style.display = "none";
        prevButton.style.display = "none";
        nextButton.style.display = "none";
        return;
    }

    let currentGraphIndex = 0;

    function updateGraph() {
        if (graphs[currentGraphIndex]) {
            graphImage.src = graphs[currentGraphIndex];
            graphImage.style.display = "block";
            console.log("Displaying graph:", graphImage.src);
        } else {
            console.warn("Invalid graph URL at index", currentGraphIndex);
            graphImage.style.display = "none";
        }

        prevButton.disabled = currentGraphIndex === 0;
        nextButton.disabled = currentGraphIndex === graphs.length - 1;
    }

    // ✅ Prevent page reload on button click
    nextButton.addEventListener("click", function (event) {
        event.preventDefault();
        if (currentGraphIndex < graphs.length - 1) {
            currentGraphIndex++;
            updateGraph();
        }
    });

    prevButton.addEventListener("click", function (event) {
        event.preventDefault();
        if (currentGraphIndex > 0) {
            currentGraphIndex--;
            updateGraph();
        }
    });

    // ✅ Handle Close button
    closeButton.addEventListener("click", function () {
        graphSection.classList.remove("show"); // Collapse the section
    });

    // ✅ Initialize first graph
    updateGraph();
});
