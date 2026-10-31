import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FlowOpt AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 FlowOpt AI")
st.subheader(
    "Intelligent Traffic Congestion Prediction and Signal Timing Optimization Using Gradient Descent"
)

st.write(
    "FlowOpt AI combines Machine Learning and Calculus to predict traffic "
    "conditions and optimize traffic signal timing."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("🚗 Traffic Inputs")

traffic_flow = st.sidebar.slider(
    "Traffic Flow (vehicles/min)",
    min_value=5,
    max_value=60,
    value=30
)

average_speed = st.sidebar.slider(
    "Average Speed (km/h)",
    min_value=5,
    max_value=60,
    value=30
)

queue_length = st.sidebar.slider(
    "Queue Length (vehicles)",
    min_value=0,
    max_value=50,
    value=15
)

current_green = st.sidebar.slider(
    "Current Green Time (seconds)",
    min_value=15,
    max_value=60,
    value=30
)

time_of_day = st.sidebar.slider(
    "Time of Day",
    min_value=0,
    max_value=23,
    value=12
)


# ============================================================
# DATA GENERATION
# ============================================================

@st.cache_data
def generate_dataset():

    np.random.seed(42)

    n = 2500

    flow = np.random.randint(5, 61, n)
    speed = np.random.randint(5, 61, n)
    queue = np.random.randint(0, 51, n)
    time = np.random.randint(0, 24, n)
    green = np.random.randint(15, 61, n)

    green_squared = green ** 2
    flow_green_ratio = flow / green

    # Synthetic traffic delay relationship
    delay = (
        10
        + 0.75 * flow
        - 0.30 * speed
        + 1.15 * queue
        + 0.08 * np.abs(time - 12)
        - 0.45 * green
        + 0.003 * green_squared
        + 18 * flow_green_ratio
        + np.random.normal(0, 5, n)
    )

    delay = np.maximum(delay, 1)

    df = pd.DataFrame({
        "Traffic_Flow": flow,
        "Average_Speed": speed,
        "Queue_Length": queue,
        "Time_of_Day": time,
        "Green_Time": green,
        "Green_Time_Squared": green_squared,
        "Flow_Green_Ratio": flow_green_ratio,
        "Delay": delay
    })

    return df


# ============================================================
# TRAIN ML MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = generate_dataset()

    features = [
        "Traffic_Flow",
        "Average_Speed",
        "Queue_Length",
        "Time_of_Day",
        "Green_Time",
        "Green_Time_Squared",
        "Flow_Green_Ratio"
    ]

    X = df[features]
    y = df["Delay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return model, df, mae, rmse, r2


model, dataset, mae, rmse, r2 = train_model()


# ============================================================
# GRADIENT CALCULATION
# ============================================================

def calculate_gradient(green_time, flow):

    """
    Derivative of the learned delay function with respect to
    green time.

    d/dg [a*g + b*g² + c*(flow/g)]

              = a + 2*b*g - c*flow/g²
    """

    coefficients = model.coef_

    # Feature order:
    # 0 = Traffic Flow
    # 1 = Average Speed
    # 2 = Queue
    # 3 = Time
    # 4 = Green Time
    # 5 = Green Time Squared
    # 6 = Flow / Green

    a = coefficients[4]
    b = coefficients[5]
    c = coefficients[6]

    gradient = (
        a
        + 2 * b * green_time
        - (c * flow) / (green_time ** 2)
    )

    return gradient


# ============================================================
# PREDICT DELAY
# ============================================================

def predict_delay(green_time):

    input_data = pd.DataFrame([{
        "Traffic_Flow": traffic_flow,
        "Average_Speed": average_speed,
        "Queue_Length": queue_length,
        "Time_of_Day": time_of_day,
        "Green_Time": green_time,
        "Green_Time_Squared": green_time ** 2,
        "Flow_Green_Ratio": traffic_flow / green_time
    }])

    return float(model.predict(input_data)[0])


# ============================================================
# GRADIENT DESCENT OPTIMIZATION
# ============================================================

def optimize_signal():

    green = float(current_green)

    learning_rate = 0.8
    iterations = 100

    history = []

    for i in range(iterations):

        gradient = calculate_gradient(
            green,
            traffic_flow
        )

        green = green - learning_rate * gradient

        # Keep signal timing realistic
        green = np.clip(green, 15, 90)

        history.append(green)

    return green, history


# ============================================================
# LIVE TRAFFIC SIMULATION
# ============================================================

def live_traffic_simulation(
    flow,
    current_green_time,
    optimized_green_time
):

    # Convert values to JavaScript safely
    flow = int(flow)
    current_green_time = int(current_green_time)
    optimized_green_time = int(round(optimized_green_time))

    html_code = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #0f1117;
    font-family: Arial, sans-serif;
    color: white;
    overflow: hidden;
}}

.container {{
    width: 100%;
    max-width: 1100px;
    margin: auto;
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background: #171a21;
    border-radius: 12px 12px 0 0;
}}

.title {{
    font-size: 21px;
    font-weight: bold;
}}

.status {{
    font-size: 16px;
    font-weight: bold;
}}

.scene {{
    position: relative;
    height: 500px;
    background: #4a4a4a;
    overflow: hidden;
    border-radius: 0 0 12px 12px;
}}

.road-horizontal {{
    position: absolute;
    left: 0;
    right: 0;
    top: 180px;
    height: 140px;
    background: #292929;
}}

.road-vertical {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 430px;
    width: 180px;
    background: #292929;
}}

.intersection {{
    position: absolute;
    left: 430px;
    top: 180px;
    width: 180px;
    height: 140px;
    background: #292929;
}}

.lane-horizontal {{
    position: absolute;
    top: 249px;
    left: 0;
    width: 100%;
    border-top: 3px dashed #d8d8d8;
}}

.lane-vertical {{
    position: absolute;
    left: 519px;
    top: 0;
    height: 100%;
    border-left: 3px dashed #d8d8d8;
}}

.stop-line-horizontal {{
    position: absolute;
    left: 420px;
    top: 180px;
    width: 8px;
    height: 140px;
    background: white;
}}

.stop-line-vertical {{
    position: absolute;
    left: 430px;
    top: 170px;
    width: 180px;
    height: 8px;
    background: white;
}}

.car {{
    position: absolute;
    width: 30px;
    height: 17px;
    border-radius: 5px;
    z-index: 10;
}}

.car::after {{
    content: "";
    position: absolute;
    width: 7px;
    height: 7px;
    background: #111;
    border-radius: 50%;
    left: 4px;
    bottom: -4px;
    box-shadow: 15px 0 #111;
}}

.car.horizontal {{
    top: 211px;
}}

.car.horizontal.reverse {{
    top: 270px;
}}

.car.vertical {{
    left: 470px;
    width: 17px;
    height: 30px;
}}

.car.vertical.reverse {{
    left: 555px;
}}

.light-box {{
    position: absolute;
    left: 385px;
    top: 105px;
    width: 65px;
    padding: 7px;
    background: #111;
    border-radius: 10px;
    z-index: 20;
}}

.light {{
    width: 17px;
    height: 17px;
    border-radius: 50%;
    margin: 5px auto;
    background: #333;
}}

.light.red.active {{
    background: #ff3131;
    box-shadow: 0 0 15px #ff3131;
}}

.light.yellow.active {{
    background: #ffd43b;
    box-shadow: 0 0 15px #ffd43b;
}}

.light.green.active {{
    background: #20e060;
    box-shadow: 0 0 15px #20e060;
}}

.info {{
    position: absolute;
    left: 15px;
    bottom: 15px;
    background: rgba(0,0,0,0.75);
    padding: 12px 16px;
    border-radius: 10px;
    z-index: 30;
    line-height: 1.7;
}}

.info b {{
    color: #66d9ff;
}}

.legend {{
    position: absolute;
    right: 15px;
    bottom: 15px;
    background: rgba(0,0,0,0.75);
    padding: 12px 16px;
    border-radius: 10px;
    z-index: 30;
}}

.dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}}

</style>

</head>


<body>

<div class="container">

    <div class="header">

        <div class="title">
            🚦 Live Traffic Simulation
        </div>

        <div class="status" id="status">
            🟢 SIMULATION STARTING
        </div>

    </div>


    <div class="scene" id="scene">

        <!-- Roads -->

        <div class="road-horizontal"></div>
        <div class="road-vertical"></div>
        <div class="intersection"></div>

        <!-- Lane markings -->

        <div class="lane-horizontal"></div>
        <div class="lane-vertical"></div>

        <!-- Stop lines -->

        <div class="stop-line-horizontal"></div>
        <div class="stop-line-vertical"></div>


        <!-- Traffic light -->

        <div class="light-box">

            <div class="light red" id="redLight"></div>
            <div class="light yellow" id="yellowLight"></div>
            <div class="light green" id="greenLight"></div>

        </div>


        <!-- Information -->

        <div class="info">

            🚗 Vehicles: <b id="vehicleCount">0</b><br>

            🚦 Signal: <b id="signalText">RED</b><br>

            ⏱️ Green Time: <b>{optimized_green_time}s</b><br>

            📊 Traffic Flow: <b>{flow} vehicles/min</b>

        </div>


        <div class="legend">

            <span class="dot" style="background:#20e060"></span>
            Green = Move
            <br>

            <span class="dot" style="background:#ff3131"></span>
            Red = Stop

        </div>

    </div>

</div>


<script>


// ============================================================
// CONFIGURATION
// ============================================================

const scene = document.getElementById("scene");

const optimizedGreen = {optimized_green_time};

const flow = {flow};


// Signal cycle
const greenDuration = 6000;

const yellowDuration = 1500;

const redDuration = 6000;


// Cars
let cars = [];

let carId = 0;


// ============================================================
// CREATE CAR
// ============================================================

function createCar(direction) {{

    const car = document.createElement("div");

    car.classList.add("car");

    const id = carId++;

    car.dataset.id = id;

    let position;

    let speed;

    if (direction === "right") {{

        car.classList.add("horizontal");

        position = -40;

        car.style.left = position + "px";

        speed = 1.5 + Math.random() * 1.5;

    }}

    else if (direction === "left") {{

        car.classList.add("horizontal");
        car.classList.add("reverse");

        position = 1120;

        car.style.left = position + "px";

        speed = 1.5 + Math.random() * 1.5;

    }}

    else if (direction === "down") {{

        car.classList.add("vertical");

        position = -40;

        car.style.top = position + "px";

        speed = 1.2 + Math.random() * 1.3;

    }}

    else {{

        car.classList.add("vertical");
        car.classList.add("reverse");

        position = 540;

        car.style.top = 540 + "px";

        speed = 1.2 + Math.random() * 1.3;

    }}


    scene.appendChild(car);

    cars.push({{
        element: car,
        direction: direction,
        position: position,
        speed: speed,
        stopped: false
    }});

}}


// ============================================================
// TRAFFIC SIGNAL
// ============================================================

let signal = "RED";

let signalStart = Date.now();


function updateSignal(){{

    const elapsed = Date.now() - signalStart;

    const cycle = greenDuration + yellowDuration + redDuration;

    const t = elapsed % cycle;


    if (t < greenDuration) {{

        signal = "GREEN";

    }}

    else if (t < greenDuration + yellowDuration) {{

        signal = "YELLOW";

    }}

    else {{

        signal = "RED";

    }}


    updateLights();

}}


// ============================================================
// LIGHT DISPLAY
// ============================================================

function updateLights(){{

    const red = document.getElementById("redLight");
    const yellow = document.getElementById("yellowLight");
    const green = document.getElementById("greenLight");

    red.classList.remove("active");
    yellow.classList.remove("active");
    green.classList.remove("active");


    if (signal === "RED") {{

        red.classList.add("active");

        document.getElementById("signalText").innerText = "RED";

        document.getElementById("status").innerText =
            "🔴 CARS STOPPING";

    }}

    else if (signal === "YELLOW") {{

        yellow.classList.add("active");

        document.getElementById("signalText").innerText = "YELLOW";

        document.getElementById("status").innerText =
            "🟡 SIGNAL CHANGING";

    }}

    else {{

        green.classList.add("active");

        document.getElementById("signalText").innerText = "GREEN";

        document.getElementById("status").innerText =
            "🟢 CARS MOVING";

    }}

}}


// ============================================================
// MOVE CARS
// ============================================================

function moveCars(){{

    for (let i = cars.length - 1; i >= 0; i--) {{

        const car = cars[i];

        let canMove = true;


        // -----------------------------------------
        // Horizontal cars
        // -----------------------------------------

        if (car.direction === "right") {{

            // Stop before intersection during red
            if (
                signal !== "GREEN" &&
                car.position > 365 &&
                car.position < 425
            ) {{

                canMove = false;

            }}

            if (canMove) {{

                car.position += car.speed;

            }}

            car.element.style.left = car.position + "px";


            if (car.position > 1120) {{

                car.element.remove();

                cars.splice(i, 1);

            }}

        }}


        else if (car.direction === "left") {{

            if (
                signal !== "GREEN" &&
                car.position < 650 &&
                car.position > 580
            ) {{

                canMove = false;

            }}

            if (canMove) {{

                car.position -= car.speed;

            }}

            car.element.style.left = car.position + "px";


            if (car.position < -50) {{

                car.element.remove();

                cars.splice(i, 1);

            }}

        }}


        // -----------------------------------------
        // Vertical cars
        // -----------------------------------------

        else if (car.direction === "down") {{

            if (car.position > 160 && car.position < 230) {{

                canMove = signal === "RED";

            }}

            if (canMove) {{

                car.position += car.speed;

            }}

            car.element.style.top = car.position + "px";


            if (car.position > 540) {{

                car.element.remove();

                cars.splice(i, 1);

            }}

        }}


        else if (car.direction === "up") {{

            if (car.position < 350 && car.position > 280) {{

                canMove = signal === "RED";

            }}

            if (canMove) {{

                car.position -= car.speed;

            }}

            car.element.style.top = car.position + "px";


            if (car.position < -50) {{

                car.element.remove();

                cars.splice(i, 1);

            }}

        }}

    }}


    document.getElementById("vehicleCount").innerText = cars.length;

}}


// ============================================================
// SPAWN TRAFFIC
// ============================================================


// More flow = more vehicles
const spawnDelay = Math.max(
    350,
    1800 - flow * 25
);


setInterval(() => {{

    createCar("right");

}}, spawnDelay);


setInterval(() => {{

    if (Math.random() > 0.35) {{

        createCar("left");

    }}

}}, spawnDelay + 400);


setInterval(() => {{

    if (Math.random() > 0.45) {{

        createCar("down");

    }}

}}, spawnDelay + 700);


setInterval(() => {{

    if (Math.random() > 0.5) {{

        createCar("up");

    }}

}}, spawnDelay + 900);


// ============================================================
// ANIMATION LOOP
// ============================================================

function animationLoop(){{

    updateSignal();

    moveCars();

    requestAnimationFrame(animationLoop);

}}


// Start
animationLoop();


</script>

</body>

</html>
"""

    components.html(
        html_code,
        height=550,
        scrolling=False
    )


# ============================================================
# MAIN ANALYSIS BUTTON
# ============================================================

st.markdown("---")

if st.button(
    "🚦 Analyze Traffic & Optimize Signal",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # ML PREDICTION
    # ========================================================

    current_delay = predict_delay(current_green)

    # ========================================================
    # GRADIENT DESCENT
    # ========================================================

    optimized_green, gradient_history = optimize_signal()

    optimized_green = round(optimized_green, 2)

    optimized_delay = predict_delay(optimized_green)

    improvement = (
        (current_delay - optimized_delay)
        / abs(current_delay)
    ) * 100


    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")

    st.header("📊 Traffic Analysis Results")


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Predicted Delay",
            f"{current_delay:.2f}"
        )

    with col2:

        st.metric(
            "Optimized Delay",
            f"{optimized_delay:.2f}"
        )

    with col3:

        st.metric(
            "Current Green",
            f"{current_green}s"
        )

    with col4:

        st.metric(
            "Optimized Green",
            f"{optimized_green}s"
        )


    st.success(
        f"🚦 Gradient Descent recommends approximately "
        f"**{optimized_green} seconds** of green time."
    )


    # ========================================================
    # BEFORE VS AFTER
    # ========================================================

    st.subheader("📈 Before vs After Optimization")

    comparison = pd.DataFrame({
        "Condition": [
            "Before Optimization",
            "After Optimization"
        ],
        "Green Time (seconds)": [
            current_green,
            optimized_green
        ],
        "Predicted Delay": [
            current_delay,
            optimized_delay
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader("🤖 Machine Learning Model Performance")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with m2:
        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with m3:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )


    # ========================================================
    # GRADIENT DESCENT GRAPH
    # ========================================================

    st.subheader("🧮 Gradient Descent Optimization")

    fig1, ax1 = plt.subplots()

    ax1.plot(
        range(1, len(gradient_history) + 1),
        gradient_history
    )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Green Time (seconds)")
    ax1.set_title("Gradient Descent Convergence")

    ax1.grid(True, alpha=0.3)

    st.pyplot(fig1)


    # ========================================================
    # DELAY COMPARISON GRAPH
    # ========================================================

    st.subheader("📊 Traffic Delay Comparison")

    fig2, ax2 = plt.subplots()

    ax2.bar(
        ["Before", "After"],
        [current_delay, optimized_delay]
    )

    ax2.set_ylabel("Predicted Delay")
    ax2.set_title("Effect of Signal Optimization")

    st.pyplot(fig2)


    # ========================================================
    # TRAFFIC DATA SAMPLE
    # ========================================================

    st.subheader("📋 Traffic Dataset Sample")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    csv_data = dataset.to_csv(index=False)

    st.download_button(
        "⬇️ Download Traffic Dataset",
        data=csv_data,
        file_name="flowopt_traffic_dataset.csv",
        mime="text/csv"
    )


    # ========================================================
    # LIVE TRAFFIC SIMULATION
    # ========================================================

    st.markdown("---")

    st.header("🚗 Live Traffic Simulation")

    st.write(
        "The optimized signal timing is now applied to a simulated "
        "traffic intersection. Vehicles move continuously, stop at "
        "red signals, and move during green signals."
    )

    st.info(
        f"🚦 Simulation is using the optimized green time: "
        f"**{optimized_green} seconds**"
    )

    live_traffic_simulation(
        traffic_flow,
        current_green,
        optimized_green
    )


    # ========================================================
    # PROJECT EXPLANATION
    # ========================================================

    st.markdown("---")

    st.subheader("🧠 How FlowOpt AI Works")

    st.markdown(
        """
        **1️⃣ Traffic Data**

        Traffic flow, average speed, queue length, time of day and
        current signal timing are used as input.

        **2️⃣ Machine Learning**

        A Linear Regression model learns the relationship between
        traffic conditions, signal timing and predicted delay.

        **3️⃣ Calculus**

        The derivative of the learned delay function with respect
        to green time is calculated.

        **4️⃣ Gradient Descent**

        The signal timing is repeatedly updated using:

        **θₙₑw = θₒₗd − α × ∇J(θ)**

        where α is the learning rate and ∇J is the gradient.

        **5️⃣ Optimization**

        Gradient Descent searches for a green-light duration that
        reduces predicted traffic delay.

        **6️⃣ Simulation**

        The optimized signal timing is applied to the live
        browser-based traffic simulation.
        """
    )


    # ========================================================
    # PROJECT NOTE
    # ========================================================

    st.caption(
        "Note: The traffic data and vehicle movement are simulated "
        "for academic demonstration. A real-world deployment would "
        "require validated traffic datasets, safety constraints and "
        "field testing."
    )