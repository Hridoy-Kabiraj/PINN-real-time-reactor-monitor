// arduino_ultrasonic_sensor.ino

// Define the pins for the ultrasonic sensor
const int trigPin = 9;
const int echoPin = 10;

// Variables for duration and distance
long duration;
int distanceCm; // Distance in centimeters

void setup() {
  // Initialize serial communication at a high baud rate for better data throughput
  // Make sure this baud rate matches the one in your Julia script
  Serial.begin(115200); 
  
  // Define pin modes
  pinMode(trigPin, OUTPUT); // Trig pin as an output
  pinMode(echoPin, INPUT);  // Echo pin as an input
}

void loop() {
  // Clear the trigPin by setting it LOW for 2 microseconds
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Set the trigPin HIGH for 10 microseconds to send a pulse
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Measure the duration of the pulse on the echoPin
  // pulseIn() returns the length of the pulse in microseconds
  duration = pulseIn(echoPin, HIGH);
  
  // Calculate the distance in centimeters
  // Speed of sound in air is approximately 340 m/s or 0.034 cm/microsecond
  // We divide by 2 because the sound travels to the object and back
  distanceCm = duration * 0.034 / 2;
  
  // --- IMPORTANT FOR JULIA COMMUNICATION ---
  // Print ONLY the numerical distance value followed by a newline character.
  // Do NOT include any descriptive text like "Distance: " or " cm".
  // This simplifies parsing on the Julia side.
  Serial.println(distanceCm);
  
  // Add a small delay to prevent overwhelming the serial port
  // Adjust this delay based on how frequently you need updates
  delay(50); 
}
