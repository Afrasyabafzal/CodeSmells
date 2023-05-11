from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# specify the path to your saved model
model_path = "./codebert-finetuned"

# load the saved model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# predict
NoSmell = """
public void performAuthentication(String username, String password) {
  if (StringUtils.isNotBlank(username) && StringUtils.isNotBlank(password)) {
    User user = userRepository.findByUsername(username);
    if (user != null) {
      String hashedPassword = HashUtils.hash(password);
      if (hashedPassword.equals(user.getPassword())) {
        String authToken = TokenUtils.generateToken(user);
        HttpSession session = request.getSession();
        session.setAttribute("authToken", authToken);
        session.setAttribute("user", user);
        response.sendRedirect("/home");
      } else {
        response.sendRedirect("/login?error=invalidPassword");
      }
    } else {
      response.sendRedirect("/login?error=invalidUsername");
    }
  } else {
    response.sendRedirect("/login?error=missingCredentials");
  }
}
 if (order.isValid()) {
    if (order.isInStock()) {
      if (order.isPaid()) {
        if (order.isShipped()) {
          // log success message
        } else {
          order.ship();
          // log success message
        }
      } else {
        order.pay();
        // log success message
      }
    } else {
      // log out-of-stock error
    }
  } else {
    // log invalid order error
  }
"""
print(NoSmell)

inputs = tokenizer(NoSmell, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# check if the input has two dimensions
if len(inputs["input_ids"].shape) == 1:
    inputs["input_ids"] = torch.unsqueeze(inputs["input_ids"], 0)
    inputs["attention_mask"] = torch.unsqueeze(inputs["attention_mask"], 0)

# Make prediction
outputs = model(**inputs)

# Get predicted class and probabilities
predicted_class = outputs.logits.argmax().item()
probs = outputs.logits.softmax(dim=1).detach().numpy()[0]

print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probs}")
