import streamlit as st

class ResourceModule:
    """
    Resource Module for providing mental health resources, blogs, and booking doctor appointments.
    """
    def __init__(self, db):
        """Initialize the Resource Module."""
        self.db = db

    def render(self):
        """Render the resource interface."""
        # Add custom CSS for styling
        self._add_css()

        st.markdown("<div class='main-header'>Resource Library</div>", unsafe_allow_html=True)

        # Section 1: Four Horizontal Boxes for Mental Health Categories
        st.markdown("<div class='sub-header'>Explore Resources by Category</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                """
                <div class='resource-box'>
                <strong>Depression</strong><br>
                Learn about coping strategies, therapies, and support groups.
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class='resource-box'>
                <strong>Anxiety</strong><br>
                Discover techniques to manage anxiety and panic attacks.
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div class='resource-box'>
                <strong>Stress</strong><br>
                Find tools and resources to reduce stress and improve well-being.
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                """
                <div class='resource-box'>
                <strong>Mindfulness</strong><br>
                Practice mindfulness exercises and meditation techniques.
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Section 2: Blog Section
        st.markdown("<div class='sub-header'>Mental Health Blogs</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='blog-box'>
            <strong>Understanding Depression: Causes and Treatments</strong><br>
            Depression is more than just feeling sad. It can affect how you think, feel, and handle daily activities. 
            Read more about its causes, symptoms, and treatment options in this detailed blog post.
            <a href="https://www.depressionlookslikeme.com/our-stories/" target="_blank">Read More</a>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class='blog-box'>
            <strong>Managing Anxiety: Tips and Techniques</strong><br>
            Anxiety can be overwhelming, but there are effective ways to manage it. Learn practical strategies to cope with anxiety in everyday life.
            <a href="https://www.rethink.org/news-and-stories/blogs/2022/07/living-with-anxiety-and-panic-attacks-beckis-story/" target="_blank">Read More</a>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Section 3: Book Doctors Option
        st.markdown("<div class='sub-header'>Book an Appointment with a Mental Health Professional</div>", unsafe_allow_html=True)
        if st.button("Schedule Appointment"):
            st.session_state.show_booking_form = True
        
        if st.session_state.get("show_booking_form", False):
            self._render_booking_form()

    def _render_booking_form(self):
        """Render the booking form for scheduling an appointment."""
        st.markdown("<div class='sub-header'>Appointment Booking Form</div>", unsafe_allow_html=True)
        
        # Hardcoded list of doctors
        doctors = [
            {"name": "Dr. Alice Johnson", "specialty": "Psychiatrist"},
            {"name": "Dr. Robert Smith", "specialty": "Therapist"},
            {"name": "Dr. Emily Davis", "specialty": "Counselor"},
            {"name": "Dr. Michael Brown", "specialty": "Clinical Psychologist"},
            {"name": "Dr. Sarah Wilson", "specialty": "Child Psychologist"},
            {"name": "Dr. David Lee", "specialty": "Addiction Specialist"},
            {"name": "Dr. Laura Martinez", "specialty": "Family Therapist"},
            {"name": "Dr. James Taylor", "specialty": "Trauma Specialist"},
            {"name": "Dr. Karen Anderson", "specialty": "Behavioral Therapist"},
            {"name": "Dr. Mark Thompson", "specialty": "Mindfulness Coach"}
        ]

        # Dropdown to select a doctor
        selected_doctor = st.selectbox(
            "Select Doctor", 
            [f"{doc['name']} - {doc['specialty']}" for doc in doctors],
            key="select_doctor"  # Unique key for the selectbox
        )

        # Date input for appointment date
        appointment_date = st.date_input("Select Date", key="appointment_date")

        # Time input for appointment time
        appointment_time = st.time_input("Select Time", key="appointment_time")

        # Text input for patient name
        patient_name = st.text_input("Your Name", key="patient_name")

        # Text input for contact number
        contact_number = st.text_input("Contact Number", key="contact_number")

        # Text area for additional notes
        additional_notes = st.text_area("Additional Notes", key="additional_notes")

        # Button to confirm the appointment
        if st.button("Confirm Appointment", key="confirm_appointment"):
            if not patient_name or not contact_number:
                st.error("Please fill in all required fields (Name and Contact Number).")
            else:
                st.success(f"Appointment booked successfully with {selected_doctor.split(' - ')[0]} on {appointment_date} at {appointment_time}.")
                st.session_state.show_booking_form = False

    def _add_css(self):
        """Add custom CSS for styling."""
        st.markdown("""
        <style>
            /* Main theme colors */
            :root {
                --background-color: #1E1E2E; /* Black background */
                --text-color: #ffffff; /* White text */
                --box-background: #ffffff; /* White box background */
                --box-text-color: #000000; /* Black text inside boxes */
                --accent-color: #6a98f0; /* Accent color for headers */
                --border-radius: 10px;
                --padding: 20px;
            }

            /* Global styles */
            .stApp {
                background-color: var(--background-color);
                color: var(--text-color);
            }

            /* Headers */
            .main-header {
                font-size: 2.5rem;
                color: var(--accent-color);
                margin-bottom: 1rem;
                font-weight: bold;
            }
            .sub-header {
                font-size: 1.5rem;
                color: var(--accent-color);
                margin-bottom: 1rem;
                font-weight: 600;
            }

            /* Resource boxes */
            .resource-box {
                background-color: var(--box-background);
                color: var(--box-text-color);
                padding: var(--padding);
                border-radius: var(--border-radius);
                text-align: center;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }

            /* Blog boxes */
            .blog-box {
                background-color: var(--box-background);
                color: var(--box-text-color);
                padding: var(--padding);
                border-radius: var(--border-radius);
                margin-bottom: 1rem;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }

            /* Buttons */
            .stButton>button {
                background-color: var(--accent-color);
                color: var(--text-color);
                border: none;
                border-radius: var(--border-radius);
                padding: 10px 20px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #8a4fff;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }

            /* Input fields */
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea {
                background-color: var(--box-background);
                color: var(--box-text-color);
                border: 1px solid var(--accent-color);
                border-radius: var(--border-radius);
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

# Initialize session state for booking form
if 'show_booking_form' not in st.session_state:
    st.session_state.show_booking_form = False