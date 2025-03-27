import streamlit as st

def display_appointment_form():
    st.title("Schedule an Appointment")
    
    # Hardcoded list of doctors
    doctors = [
        "Dr. Smith",
        "Dr. Johnson",
        "Dr. Williams",
        "Dr. Brown",
        "Dr. Jones",
        "Dr. Garcia",
        "Dr. Martinez",
        "Dr. Davis",
        "Dr. Rodriguez",
        "Dr. Wilson"
    ]
    
    # Appointment form
    with st.form("appointment_form"):
        st.selectbox("Select Doctor", doctors)
        st.date_input("Select Date")
        st.time_input("Select Time")
        st.text_area("Additional Notes")
        
        submitted = st.form_submit_button("Book Appointment")
        if submitted:
            st.success("Appointment booked successfully!")

if __name__ == "__main__":
    display_appointment_form()
