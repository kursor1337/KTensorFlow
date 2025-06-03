import Foundation

enum Environment: String {
    case debug
    case release
    
    var isRelease: Bool { self == .release }
}

final class EnvironmentService {
    private enum Constants {
        static let buildConfigurationNameKey = "BUILD_CONFIGURATION_NAME"
    }
    
    static let shared = EnvironmentService()
    static var isRelease: Bool { shared.getCurrentEnvironment().isRelease }
    
    private(set) lazy var currentEnvironment = getCurrentEnvironment()
    
    private init() {}
    
    private func getCurrentEnvironment() -> Environment {
        guard
            let instance = Bundle.main.object(forInfoDictionaryKey: Constants.buildConfigurationNameKey) as? String,
            let environment = Environment(rawValue: instance)
        else {
            return .debug
        }
        
        return environment
    }
}
