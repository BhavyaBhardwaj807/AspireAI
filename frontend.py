import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import datetime





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch




st.set_page_config(page_title="AspireAI: \n Your AI Career Assistant")

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Ancizar+Sans:ital,wght@0,100..1000;1,100..1000&display=swap" rel="stylesheet">
  <h1 style="
    font-family: 'Ancizar Sans', sans-serif;
    background: linear-gradient(90deg, #1E90FF, #FF1493);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 48px;
    text-align: center;
    font-weight: 800;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.1);
">
    ✨ Aspire<span style='color:#FF1493;'>AI</span> 🚀
</h1>
<p style="text-align:center; font-size: 20px; color: gray;">
    🧠 Empowering your career journey with intelligence
</p>
<p style="text-align:center; font-size: 16px; color: #888; margin-top: -10px;">
    Your personalized AI career coach — guiding your path, step by step.
</p>

""",
    unsafe_allow_html=True
)

menu = ["Profile & Progress", "Career Path Recommendations", "Skill Gap Analysis", "Network Readiness", "Career Time Capsule"]
choice = st.sidebar.selectbox("Go to", menu)

# ----------------------------- PROFILE & PROGRESS SECTION -----------------------------
if choice == "Profile & Progress":
    st.markdown("<h2 style='text-align: center;'>Profile and Progress Tracking</h2>", unsafe_allow_html=True)

    # ✅ Always initialize session state values before usage
    if "profile_data" not in st.session_state:
        st.session_state.profile_data = {
            "name": "",
            "education": "",
            "skills": "",
            "interests": "",
            "goals": ""
        }

    if "completed_skills" not in st.session_state:
        st.session_state.completed_skills = []

    # ✅ Alias for cleaner code
    profile_data = st.session_state.profile_data

    # 🧾 User Profile Form
    with st.form("user_profile_form"):
        name = st.text_input("Your Name", profile_data["name"])
        education = st.text_input("Your Education (e.g., B.Tech in CSE)", profile_data["education"])
        skills = st.text_area("List Your Skills (comma-separated)", profile_data["skills"])
        interests = st.text_area("Your Interests (e.g., AI, Web Dev, Data Science)", profile_data["interests"])
        goals = st.text_area("Your Career Goals", profile_data["goals"])
        submitted = st.form_submit_button("Save Profile")

    # 💾 Save to session state
    if submitted:
        st.session_state.profile_data = {
            "name": name,
            "education": education,
            "skills": skills,
            "interests": interests,
            "goals": goals
        }
        st.session_state.completed_skills = []  # Reset
        st.success("✅ Profile Saved Successfully!")

    # 📋 Show profile data safely
    profile_data = st.session_state.profile_data
    st.markdown(f"**Name:** {profile_data.get('name', 'Not provided')}")
    st.markdown(f"**Education:** {profile_data.get('education', 'Not provided')}")
    st.markdown(f"**Skills:** {profile_data.get('skills', 'Not provided')}")
    st.markdown(f"**Interests:** {profile_data.get('interests', 'Not provided')}")
    st.markdown(f"**Career Goals:** {profile_data.get('goals', 'Not provided')}")

    skill_list = [skill.strip() for skill in st.session_state.profile_data.get("skills", "").split(",") if skill.strip()]

    if skill_list:
        st.subheader("🌟 Skill Progress Tracker")
        updated_completed = []

        for skill in skill_list:
            checked = st.checkbox(skill, value=(skill in st.session_state.completed_skills))
            if checked:
                updated_completed.append(skill)

        # Update session state with current completed skills
        st.session_state.completed_skills = updated_completed

        # Show progress bar and stats
        percent_complete = int((len(updated_completed) / len(skill_list)) * 100)
        st.progress(percent_complete)
        st.markdown(f"**Skills Completed: {len(updated_completed)} / {len(skill_list)} ({percent_complete}%)**")
    else:
        st.info("📜 Add skills in your profile to start tracking progress.")

# ----------------------------- CAREER PATH SECTION -----------------------------
if choice == "Career Path Recommendations":
    st.markdown("<h2 style='text-align: center;'>Career Path Recommendations</h2>", unsafe_allow_html=True)

    with open("cleaned_career_knowledge_base.json", "r") as f:
        job_skills = json.load(f)

    # User skill input
    user_input = st.text_area("Enter your skills (comma-separated)")
    user_skills = [skill.strip().lower() for skill in user_input.split(",") if skill.strip()]

    if st.button("Suggest Career Paths") and user_skills:
        # ✅ Save to session and profile
        st.session_state['user_skills'] = user_skills
        profile_data = {"skills": ", ".join(user_skills)}
        with open("profile.json", "w") as f:
            json.dump(profile_data, f)

        scores = []
        match_details = {}

        for role, required_skills in job_skills.items():
            req_set = set(required_skills)
            user_set = set(user_skills)

            matched = user_set & req_set
            missing = req_set - user_set
            match_ratio = len(matched) / len(req_set) if req_set else 0

            scores.append((role, match_ratio))
            match_details[role] = {"match": matched, "missing": missing}

        # Sort and get top 5
        top_roles = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

        # Bar chart
        df = pd.DataFrame(top_roles, columns=["Role", "Match %"])
        df["Match %"] = df["Match %"] * 100  # convert to %
        st.subheader("📊 Match Score Chart")
        st.bar_chart(data=df.set_index("Role"))

        # Detailed breakdown
        st.subheader("🧠 Recommendations & Missing Skills")
        for role, score in top_roles:
            percent = round(score * 100)
            matched = match_details[role]["match"]
            missing = match_details[role]["missing"]

            st.markdown(f"### 🔹 {role.title()} — {percent}% Match")
            st.markdown(f"- ✅ **Matched Skills:** {', '.join(sorted(matched)) if matched else 'None'}")
            st.markdown(f"- ❌ **Missing Skills to Learn:** {', '.join(sorted(missing)) if missing else 'None'}")
            st.markdown("---")
    else:
        st.info("Please enter at least one skill to get recommendations.")
# ----------------------------- OTHER SECTIONS -----------------------------
def load_profile():
    if not os.path.exists("profile.json"):
        return set()
    with open("profile.json", "r") as f:
        data = json.load(f)
    skills_str = data.get("skills", "")
    skills_set = set(skill.strip().lower() for skill in skills_str.split(",") if skill.strip())
    return skills_set

def save_profile(skills_set):
    with open("profile.json", "w") as f:
        json.dump({"skills": ", ".join(sorted(skills_set))}, f)

def get_user_skills():
    if 'user_skills' in st.session_state and st.session_state['user_skills']:
        return st.session_state['user_skills']
    else:
        skills = load_profile()
        if skills:
            st.session_state['user_skills'] = skills
            return skills
        else:
            st.warning("⚠️ No skills found. Please add your skills first.")
            st.stop()

def load_career_knowledge():
    if not os.path.exists("career_knowledge_base.json"):
        st.error("⚠️ career_knowledge_base.json file not found!")
        st.stop()
    with open("career_knowledge_base.json", "r") as f:
        return json.load(f)

def skill_gap(user_skills, required_skills):
    matched = user_skills & required_skills
    missing = required_skills - user_skills
    return matched, missing

def show_skill_gap_analysis(matched, missing):
    st.subheader("📊 Skill Gap Analysis")
    st.markdown(f"**✅ Matched Skills:** {', '.join(sorted(matched)) if matched else 'None'}")
    st.markdown(f"**🚫 Missing Skills:** {', '.join(sorted(missing)) if missing else 'None'}")

    labels = ["Matched Skills", "Missing Skills"]
    values = [len(matched), len(missing)]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["green", "red"])
    ax.set_ylabel("Skill Count")
    st.pyplot(fig)

def show_learning_resources(missing):
    st.subheader("📚 Suggested YouTube Resources")
    youtube_resources = {
        "python": "https://www.youtube.com/watch?v=gfDE2a7MKjA (CodeWithHarry)",
        "sql": "https://www.youtube.com/watch?v=hlGoQC332VM (CodeBasics)",
        "django": "https://www.youtube.com/watch?v=JxzZxdht-XY (Telusko)",
        "javascript": "https://www.youtube.com/playlist?list=PLfqMhTWNBTe0b2nM6JHVCnAkhQRGiZMSJ (Apna College)",
        "html": "https://www.youtube.com/watch?v=BsDoLVMnmZs (CodeWithHarry)",
        "css": "https://www.youtube.com/watch?v=ESnrn1kAD4E (Apna College)",
        "java": "https://www.youtube.com/playlist?list=PLdo5W4Nhv31a8UcTrt0oKgoI1oS6N7-3y (Jenny’s Lectures)",
        "machine learning": "https://www.youtube.com/watch?v=Gv9_4yMHFhI (Krish Naik)",
        "deep learning": "https://www.youtube.com/watch?v=aircAruvnKk (Krish Naik)",
        "pandas": "https://www.youtube.com/watch?v=vmEHCJofslg (CodeBasics)",
        "power bi": "https://www.youtube.com/watch?v=AGrl-H87pRU (Learn With Nandeshwar)",
        "tableau": "https://www.youtube.com/watch?v=9I9DtFOVPIg (Great Learning)",
        "excel": "https://www.youtube.com/c/ExcelSuperstar (Excel Superstar)",
        "numpy": "https://www.youtube.com/watch?v=QUT1VHiLmmI (CodeBasics)",
        "c++": "https://www.youtube.com/watch?v=z9bZufPHFLU (Apna College)"
    }

    for skill in sorted(missing):
        if skill in youtube_resources:
            st.markdown(f"- 🎥 **{skill.title()}**: [Watch Resource]({youtube_resources[skill]})")
        else:
            st.markdown(f"- ❓ **{skill.title()}**: No resource available yet.")



# --------- Main code block ---------



if choice == "Skill Gap Analysis":
    st.markdown("<h2 style='text-align: center;'>Skill Gap & Personalized Resources</h2>", unsafe_allow_html=True)

    user_skills = get_user_skills()
    st.markdown("### 📘 Your Current Skills")
    st.write(user_skills)

    role_skills = load_career_knowledge()

    selected_role = st.selectbox("🎯 Select a Career Role", list(role_skills.keys()), key="skill_gap_role")
    st.write("🎯 Selected Role:", selected_role)

    required_skills = set(skill.lower() for skill in role_skills[selected_role])
    matched, missing = skill_gap(set(user_skills), required_skills)
    st.write("✅ Matched:", matched)
    st.write("❌ Missing:", missing)

    show_skill_gap_analysis(matched, missing)
    show_learning_resources(missing)

    st.markdown("---")
    st.markdown("## Weekly Study Planner")

    selected_role_plan = st.selectbox("🎯 Select a Career Role for Study Plan", list(role_skills.keys()), key="study_plan_role")
    required_skills_plan = set(skill.lower() for skill in role_skills[selected_role_plan])
    matched_plan, missing_plan = skill_gap(set(user_skills), required_skills_plan)
    

    # Now the AI-generated study plan block, wrapped in try/except
    try:
        if missing_plan:
            st.subheader("🤖 AI-Generated Study Plan")

            if st.button("Generate with Gemini"):
                with st.spinner("Generating study plan using Gemini..."):
                    import google.generativeai as genai

                    # Secure your key via environment variable or st.secrets
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # or os.getenv("GEMINI_API_KEY")

                    skills_text = ', '.join(missing_plan)
                    prompt = (
                        f"You are an expert career coach. Create a detailed, step-by-step 7-day personalized study plan for a beginner to learn the following skills: {skills_text}.\n"
                        "Each day should include:\n"
                        "- A topic\n"
                        "- 2-3 beginner-friendly tasks or activities\n"
                        "- Use bullet points or numbered format\n"
                        "- Keep the tone encouraging and clear."
                    )

                    model = genai.GenerativeModel("models/gemini-1.5-flash")
                    response = model.generate_content(prompt)

                    st.code(response.text)

        else:
            st.info("🎉 You have all the required skills for this role!")

    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")


elif choice == "Network Readiness":
    st.markdown("<h2 style='text-align: center;'>Network Readiness</h2>", unsafe_allow_html=True)

    st.subheader("🔗 LinkedIn Profile Audit")
    linkedin_url = st.text_input("Enter your LinkedIn profile URL (optional)")
    profile_text = st.text_area("Paste your LinkedIn 'About' section or full profile text", height=250)

    if st.button("🚀 Run Audit with Gemini"):
        if profile_text.strip() == "":
            st.warning("⚠️ Please paste your LinkedIn profile content.")
        else:
            with st.spinner("Analyzing your profile with Gemini..."):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                    prompt = (
                        "You are an expert LinkedIn coach. Analyze the following LinkedIn profile content:\n"
                        f"{profile_text}\n\n"
                        "Perform a professional audit on the following areas:\n"
                        "- Headline\n"
                        "- About section\n"
                        "- Skills & Endorsements\n"
                        "- Experience section\n"
                        "- Recommendations\n"
                        "- Activity level\n\n"
                        "Use ✅ for good, ⚠️ for average, ❌ for poor or missing. Give brief feedback under each section.\n"
                        "At the end, give a profile score out of 10 (number only) based on completeness, clarity, and impact.\n"
                        "Then provide 2-3 improvement suggestions."
                    )

                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)

                    full_text = response.text

                    # Extract numeric score from response
                    import re
                    score_match = re.search(r"(\b\d{1,2}\b)(?=\s*/\s*10|\s*out of 10)", full_text)
                    score = int(score_match.group(1)) if score_match else 7

                    st.markdown("### 🧾 Audit Report")
                    st.markdown(full_text)

                    st.markdown("### 📈 Profile Strength")
                    st.progress(score / 10)

                    st.info(f"🧠 Gemini Score: **{score}/10**")

                except Exception as e:
                    st.error(f"🚨 Gemini API error: {e}")

    st.markdown("---")

    st.subheader("🎯 Personal Branding Checklist")
    checklist_items = [
        "Professional LinkedIn headline",
        "Consistent username across platforms",
        "Portfolio website or GitHub link",
        "Custom LinkedIn URL",
        "Featured posts or articles",
        "Active engagement (likes, comments on relevant posts)"
    ]
    for item in checklist_items:
        st.checkbox(item)

    st.markdown("---")

    st.subheader("📊 Network Strength Score")
    connections = st.slider("Number of LinkedIn Connections", 0, 5000, 100)
    endorsements = st.slider("Number of Skill Endorsements", 0, 100, 10)
    posts = st.slider("Number of Professional Posts in Last 3 Months", 0, 20, 2)
    comments = st.slider("Comments on Industry Content", 0, 50, 5)

    score = min(100, connections * 0.05 + endorsements * 0.5 + posts * 2 + comments * 1)
    st.progress(int(score))
    st.markdown(f"**Your Network Strength Score: {int(score)} / 100**")

    st.markdown("---")

    st.subheader("📆 Weekly Networking Challenge")
    challenges = [
        "Connect with 3 new professionals in your field.",
        "Comment meaningfully on 2 LinkedIn posts.",
        "Write a short post about something you learned this week.",
        "Reach out to an alum from your college working in your target field.",
        "Share a useful article and add your perspective.",
        "Send a thank you message to a mentor or teacher.",
        "Join a webinar or virtual meetup and post about it."
    ]
    import random
    st.info(random.choice(challenges))

    st.markdown("---")

    


elif choice == "Career Time Capsule":
    st.markdown("<h2 style='text-align: center;'>🕰️ Career Time Capsule</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Reflect on your present and dream big for your future. We'll help you revisit this moment later.</p>", unsafe_allow_html=True)
    
    with st.form("capsule_form"):
        st.markdown("### 📍 Today – Your Current Self")
        current_summary = st.text_area("Describe where you are in your career journey right now.", height=150)

        st.markdown("### 🚀 Future You – Vision for 1 Year Later")
        future_goals = st.text_area("What do you want to achieve in the next year?", height=150)

        st.markdown("### Message to Future You")
        advice_to_future = st.text_area("Write a message or advice to your future self.", height=150)

        st.markdown("### ⏳ Reveal Year")
        reveal_year = st.selectbox("Choose when you'd like to revisit this:", [2025, 2026, 2027, 2028, 2029])

        submitted = st.form_submit_button("📦 Lock My Time Capsule")

    if submitted:
        st.success("✅ Your Career Time Capsule has been locked safely!")
        st.balloons()

        with st.expander("🔐 View Your Capsule Now"):
            st.markdown(f"""
            ### 🕰️ Career Time Capsule ({reveal_year})
            **📍 Today:**  
            {current_summary}

            **🚀 Future Goals:**  
            {future_goals}

            **💌 Message to Future Self:**  
            {advice_to_future}
            """)

        # Optional: let user download as .txt
        capsule_text = f"""
        Career Time Capsule ({reveal_year})

        📍 Current Self:
        {current_summary}

        🚀 Goals for the Future:
        {future_goals}

        💌 Message to Future You:
        {advice_to_future}
        """

        st.download_button("📥 Download Capsule", data=capsule_text, file_name="career_time_capsule.txt", mime="text/plain")
