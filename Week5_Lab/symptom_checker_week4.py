#!/usr/bin/env python3
"""
AI 250 - Week 4: Interactive Disease Prediction System

This program loads a trained disease prediction model and allows users to
interactively check symptoms and get disease predictions with confidence scores.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import sys


class SymptomChecker:
    def __init__(self, data_path=""):
        """Initialize the symptom checker by loading and training the model."""
        self.data_path = data_path
        self.model = None
        self.symptoms_list = []
        self.diseases_list = []
    
    def load_and_train_model(self):
        """Load training data and train the Random Forest model."""
        print("Loading training data...")
        
        # Check if data files exist
        train_file = os.path.join(self.data_path, "training_data.csv")
        if not os.path.exists(train_file):
            print(f"Error: Training data not found at {train_file}")
            print("Please ensure the Lab4_DiseasePrediction folder contains training_data.csv")
            sys.exit(1)
        
        try:
            # Load training data
            train_data = pd.read_csv(train_file)
            
            # Clean up column names (remove any unnamed columns from trailing commas)
            train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
            
            # Extract features and labels
            X_train = train_data.drop('prognosis', axis=1)
            y_train = train_data['prognosis']
            
            # Store symptom names and disease names
            self.symptoms_list = list(X_train.columns)
            self.diseases_list = sorted(y_train.unique())
            
            print(f"Loaded {len(train_data)} training samples")
            print(f"{len(self.symptoms_list)} symptoms available")
            print(f"{len(self.diseases_list)} diseases in database")
            
            # Train Random Forest model (best performer from Lab 4)
            print("Training Random Forest model...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            print("Model trained successfully!\n")
            
        except Exception as e:
            print(f"Error loading or training model: {e}")
            sys.exit(1)
    
    def normalize_symptom(self, symptom_input):
        """Normalize user input to match symptom names in the dataset."""
        # Convert to lowercase and replace spaces with underscores
        normalized = symptom_input.lower().strip().replace(' ', '_').replace('-', '_')
        return normalized
    
    def find_matching_symptoms(self, user_input):
        """Find symptoms that match the user input."""
        normalized_input = self.normalize_symptom(user_input)
        matches = []
        
        for symptom in self.symptoms_list:
            if normalized_input in symptom.lower() or symptom.lower() in normalized_input:
                matches.append(symptom)
        
        return matches
    
    def create_symptom_vector(self, selected_symptoms):
        """Create a binary vector for the selected symptoms."""
        symptom_vector = np.zeros(len(self.symptoms_list))
        
        for symptom in selected_symptoms:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                symptom_vector[idx] = 1
        
        return symptom_vector.reshape(1, -1)
    
    def predict_disease(self, selected_symptoms):
        """Predict diseases based on selected symptoms with confidence scores."""
        if not selected_symptoms:
            print("No symptoms selected. Please add symptoms first.")
            return
        
        # Create symptom vector
        symptom_vector = self.create_symptom_vector(selected_symptoms)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(symptom_vector)[0]
        
        # Get the predicted class
        prediction = self.model.predict(symptom_vector)[0]
        
        # Create a list of (disease, probability) tuples and sort by probability
        disease_probabilities = list(zip(self.model.classes_, probabilities))
        disease_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        print("\n" + "="*70)
        print("DISEASE PREDICTION RESULTS")
        print("="*70)
        
        print(f"\n✓ Primary Diagnosis: {prediction}")
        print(f"✓ Confidence: {max(probabilities)*100:.2f}%")
        
        print("\nTop 5 Possible Conditions (with confidence scores):")
        print("-"*70)
        
        for i, (disease, prob) in enumerate(disease_probabilities[:5], 1):
            confidence = prob * 100
            # Create a simple progress bar
            bar_length = int(confidence / 2)  # Scale to 50 chars max
            bar = "=" * bar_length + "-" * (50 - bar_length)
            print(f"{i}. {disease}")
            print(f"   [{bar}] {confidence:.2f}%")
            print()
        
        print("="*70)
        print("DISCLAIMER: This is an AI prediction tool for educational purposes.")
        print("   Always consult a qualified healthcare professional for medical advice.")
        print("="*70 + "\n")
    
    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "="*70)
        print("SYMPTOM CHECKER - MAIN MENU")
        print("="*70)
        print("1. Add symptom")
        print("2. List available symptoms")
        print("3. View current symptoms")
        print("4. Remove symptom")
        print("5. Clear all symptoms")
        print("6. Check diagnosis (predict disease)")
        print("7. Exit")
        print("="*70)
    
    def list_symptoms(self, filter_text=""):
        """Display all available symptoms, optionally filtered."""
        if filter_text:
            symptoms = [s for s in self.symptoms_list if filter_text.lower() in s.lower()]
            print(f"\n📋 Symptoms matching '{filter_text}':", flush=True)
        else:
            symptoms = self.symptoms_list
            print(f"\n📋 All available symptoms ({len(symptoms)} total):", flush=True)
        
        print("-"*70, flush=True)
        
        # Display in columns for better readability
        for i in range(0, len(symptoms), 2):
            left = f"{i+1}. {symptoms[i]}"
            if i+1 < len(symptoms):
                right = f"{i+2}. {symptoms[i+1]}"
                print(f"{left:<40} {right}", flush=True)
            else:
                print(left, flush=True)
    
    
    def run(self):
        """Run the interactive symptom checker."""
        print("\n" + "="*70)
        print("AI 250 - INTERACTIVE DISEASE PREDICTION SYSTEM")
        print("="*70)
        print()
        
        # Load and train model
        self.load_and_train_model()

        
        # Store user's selected symptoms
        current_symptoms = []
        
        # Main interaction loop
        while True:
            self.display_menu()
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "1":
                # Add symptom
                print("\nADD SYMPTOM")
                print("-"*70)
                symptom_input = input("Enter symptom (or part of symptom name): ").strip()
                
                if not symptom_input:
                    print("No input provided.")
                    continue
                
                # Find matching symptoms
                matches = self.find_matching_symptoms(symptom_input)
                
                if not matches:
                    print(f"No symptoms found matching '{symptom_input}'")
                    print("Tip: Try using keywords like 'fever', 'pain', 'cough', etc.")
                elif len(matches) == 1:
                    symptom = matches[0]
                    if symptom not in current_symptoms:
                        current_symptoms.append(symptom)
                        print(f"Added: {symptom}")
                    else:
                        print(f"Symptom already added: {symptom}")
                else:
                    print(f"Found {len(matches)} matching symptoms:")
                    for i, symptom in enumerate(matches, 1):
                        status = "✓ (added)" if symptom in current_symptoms else ""
                        print(f"{i}. {symptom} {status}")
                    
                    try:
                        selection = input(f"\nSelect symptom number (1-{len(matches)}) or 0 to cancel: ").strip()
                        if selection == "0":
                            continue
                        idx = int(selection) - 1
                        if 0 <= idx < len(matches):
                            symptom = matches[idx]
                            if symptom not in current_symptoms:
                                current_symptoms.append(symptom)
                                print(f"✅ Added: {symptom}")
                            else:
                                print(f"Symptom already added: {symptom}")
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Invalid input.")
            
            elif choice == "2":
                # List available symptoms
                print("\nLIST SYMPTOMS")
                print("-"*70)
                filter_option = input("Filter symptoms (press Enter to see all): ").strip()
                self.list_symptoms(filter_option)
            
            elif choice == "3":
                # View current symptoms
                print("\nYOUR CURRENT SYMPTOMS")
                print("-"*70)
                if current_symptoms:
                    for i, symptom in enumerate(current_symptoms, 1):
                        print(f"{i}. {symptom}")
                    print(f"\nTotal: {len(current_symptoms)} symptom(s)")
                else:
                    print("No symptoms added yet.")
                print("-"*70)
            
            elif choice == "4":
                # Remove symptom
                if not current_symptoms:
                    print("\nNo symptoms to remove.")
                    continue
                
                print("\nREMOVE SYMPTOM")
                print("-"*70)
                for i, symptom in enumerate(current_symptoms, 1):
                    print(f"{i}. {symptom}")
                
                try:
                    selection = input(f"\nSelect symptom to remove (1-{len(current_symptoms)}) or 0 to cancel: ").strip()
                    if selection == "0":
                        continue
                    idx = int(selection) - 1
                    if 0 <= idx < len(current_symptoms):
                        removed = current_symptoms.pop(idx)
                        print(f"Removed: {removed}")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            
            elif choice == "5":
                # Clear all symptoms
                if current_symptoms:
                    confirm = input(f"\nClear all {len(current_symptoms)} symptoms? (y/n): ").strip().lower()
                    if confirm == 'y':
                        current_symptoms.clear()
                        print("All symptoms cleared.")
                else:
                    print("\nNo symptoms to clear.")
            
            elif choice == "6":
                # Predict disease
                if not current_symptoms:
                    print("\nPlease add at least one symptom before checking diagnosis.")
                else:
                    self.predict_disease(current_symptoms)
                    
                    # Ask if user wants to start over
                    again = input("\nCheck another set of symptoms? (y/n): ").strip().lower()
                    if again == 'y':
                        current_symptoms.clear()
                        print("Symptoms cleared. Starting fresh.")
            
            elif choice == "7":
                # Exit
                print("\n" + "="*70)
                print("Thank you for using the Disease Prediction System!")
                print("   Stay healthy and consult a doctor if you have health concerns.")
                print("="*70 + "\n")
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")


def main():
    """Main entry point for the symptom checker."""
    checker = SymptomChecker()
    try:
        checker.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
